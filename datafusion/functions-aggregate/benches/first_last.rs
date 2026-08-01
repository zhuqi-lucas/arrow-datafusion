// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use arrow::array::{ArrayRef, BooleanArray, Int64Array, ListArray, StructArray};
use arrow::buffer::OffsetBuffer;
use arrow::compute::SortOptions;
use arrow::datatypes::{DataType, Field, Fields, Float64Type, Int64Type, Schema};
use arrow::util::bench_util::{
    create_boolean_array, create_primitive_array, create_string_array_with_len,
};
use datafusion_common::instant::Instant;
use std::hint::black_box;
use std::sync::Arc;

use datafusion_expr::{
    Accumulator, AggregateUDFImpl, EmitTo, GroupsAccumulator, function::AccumulatorArgs,
};
use datafusion_functions_aggregate::first_last::{
    FirstValue, LastValue, TrivialFirstValueAccumulator, TrivialLastValueAccumulator,
};
use datafusion_physical_expr::PhysicalSortExpr;
use datafusion_physical_expr::expressions::col;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};

/// Build a `GroupsAccumulator` for an arbitrary value type, so the nested-type
/// (`Struct` / `List`) fast paths added for `first_value` / `last_value` can be
/// exercised with the same harness as the primitive ones.
fn prepare_typed_groups_accumulator(
    is_first: bool,
    value_type: DataType,
) -> Box<dyn GroupsAccumulator> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("value", value_type.clone(), true),
        Field::new("ord", DataType::Int64, true),
    ]));

    let order_expr = col("ord", &schema).unwrap();
    let sort_expr = PhysicalSortExpr {
        expr: order_expr,
        options: SortOptions::default(),
    };

    let value_field: Arc<Field> = Field::new("value", value_type, true).into();
    let accumulator_args = AccumulatorArgs {
        return_field: Arc::clone(&value_field),
        schema: &schema,
        expr_fields: &[value_field],
        ignore_nulls: false,
        order_bys: std::slice::from_ref(&sort_expr),
        is_reversed: false,
        name: if is_first {
            "FIRST_VALUE(value ORDER BY ord)"
        } else {
            "LAST_VALUE(value ORDER BY ord)"
        },
        is_distinct: false,
        exprs: &[col("value", &schema).unwrap()],
    };

    if is_first {
        FirstValue::new()
            .create_groups_accumulator(accumulator_args)
            .unwrap()
    } else {
        LastValue::new()
            .create_groups_accumulator(accumulator_args)
            .unwrap()
    }
}

fn create_trivial_accumulator(
    is_first: bool,
    ignore_nulls: bool,
) -> Box<dyn Accumulator> {
    if is_first {
        Box::new(
            TrivialFirstValueAccumulator::try_new(&DataType::Int64, ignore_nulls)
                .unwrap(),
        )
    } else {
        Box::new(
            TrivialLastValueAccumulator::try_new(&DataType::Int64, ignore_nulls).unwrap(),
        )
    }
}

#[expect(clippy::needless_pass_by_value)]
#[expect(clippy::too_many_arguments)]
fn evaluate_bench(
    c: &mut Criterion,
    is_first: bool,
    emit_to: EmitTo,
    name: &str,
    values: ArrayRef,
    ord: ArrayRef,
    opt_filter: Option<&BooleanArray>,
    num_groups: usize,
) {
    let n = values.len();
    let group_indices: Vec<usize> = (0..n).map(|i| i % num_groups).collect();
    let value_type = values.data_type().clone();

    c.bench_function(name, |b| {
        b.iter_batched(
            || {
                let mut accumulator =
                    prepare_typed_groups_accumulator(is_first, value_type.clone());
                accumulator
                    .update_batch(
                        &[Arc::clone(&values), Arc::clone(&ord)],
                        &group_indices,
                        opt_filter,
                        num_groups,
                    )
                    .unwrap();
                accumulator
            },
            |mut accumulator| {
                black_box(accumulator.evaluate(emit_to).unwrap());
            },
            BatchSize::SmallInput,
        )
    });
}

#[expect(clippy::needless_pass_by_value)]
fn update_bench(
    c: &mut Criterion,
    is_first: bool,
    name: &str,
    values: ArrayRef,
    ord: ArrayRef,
    opt_filter: Option<&BooleanArray>,
    num_groups: usize,
) {
    let n = values.len();
    let group_indices: Vec<usize> = (0..n).map(|i| i % num_groups).collect();
    let value_type = values.data_type().clone();

    // Initialize with worst-case ordering so update_batch forces rows comparison for all groups.
    let worst_ord: ArrayRef = Arc::new(Int64Array::from(vec![
        if is_first {
            i64::MAX
        } else {
            i64::MIN
        };
        n
    ]));

    c.bench_function(name, |b| {
        b.iter_batched(
            || {
                let mut accumulator =
                    prepare_typed_groups_accumulator(is_first, value_type.clone());
                accumulator
                    .update_batch(
                        &[Arc::clone(&values), Arc::clone(&worst_ord)],
                        &group_indices,
                        None, // no filter: ensure all groups are initialised
                        num_groups,
                    )
                    .unwrap();
                accumulator
            },
            |mut accumulator| {
                for _ in 0..100 {
                    #[expect(clippy::unit_arg)]
                    black_box(
                        accumulator
                            .update_batch(
                                &[Arc::clone(&values), Arc::clone(&ord)],
                                &group_indices,
                                opt_filter,
                                num_groups,
                            )
                            .unwrap(),
                    );
                }
            },
            BatchSize::SmallInput,
        )
    });
}

#[expect(clippy::needless_pass_by_value)]
fn merge_bench(
    c: &mut Criterion,
    is_first: bool,
    name: &str,
    values: ArrayRef,
    ord: ArrayRef,
    opt_filter: Option<&BooleanArray>,
    num_groups: usize,
) {
    let n = values.len();
    let group_indices: Vec<usize> = (0..n).map(|i| i % num_groups).collect();
    let is_set: ArrayRef = Arc::new(BooleanArray::from(vec![true; n]));
    let value_type = values.data_type().clone();

    // Initialize with worst-case ordering so update_batch forces rows comparison for all groups.
    let worst_ord: ArrayRef = Arc::new(Int64Array::from(vec![
        if is_first {
            i64::MAX
        } else {
            i64::MIN
        };
        n
    ]));

    c.bench_function(name, |b| {
        b.iter_batched(
            || {
                // Prebuild accumulator
                let mut accumulator =
                    prepare_typed_groups_accumulator(is_first, value_type.clone());
                accumulator
                    .update_batch(
                        &[Arc::clone(&values), Arc::clone(&worst_ord)],
                        &group_indices,
                        opt_filter,
                        num_groups,
                    )
                    .unwrap();
                accumulator
            },
            |mut accumulator| {
                for _ in 0..100 {
                    #[expect(clippy::unit_arg)]
                    black_box(
                        accumulator
                            .merge_batch(
                                &[
                                    Arc::clone(&values),
                                    Arc::clone(&ord),
                                    Arc::clone(&is_set),
                                ],
                                &group_indices,
                                num_groups,
                            )
                            .unwrap(),
                    );
                }
            },
            BatchSize::SmallInput,
        )
    });
}

#[expect(clippy::needless_pass_by_value)]
fn trivial_update_bench(
    c: &mut Criterion,
    is_first: bool,
    ignore_nulls: bool,
    name: &str,
    values: ArrayRef,
) {
    c.bench_function(name, |b| {
        b.iter_custom(|iters| {
            // The bench is way too fast, so apply scaling factor
            let mut accumulators: Vec<Box<dyn Accumulator>> = (0..iters * 100)
                .map(|_| create_trivial_accumulator(is_first, ignore_nulls))
                .collect();
            let start = Instant::now();
            for acc in &mut accumulators {
                #[expect(clippy::unit_arg)]
                black_box(acc.update_batch(&[Arc::clone(&values)]).unwrap());
            }
            start.elapsed()
        })
    });
}

/// A 3-field struct value column `(Int64, Utf8, Float64)` — the shape produced
/// by rewriting three peer `first_value(col ORDER BY o)` calls into a single
/// `first_value(named_struct(..) ORDER BY o)` (the coalesce-peers optimization).
fn create_struct_array(n: usize, null_density: f32) -> ArrayRef {
    let a = Arc::new(create_primitive_array::<Int64Type>(n, null_density)) as ArrayRef;
    let b =
        Arc::new(create_string_array_with_len::<i32>(n, null_density, 16)) as ArrayRef;
    let d = Arc::new(create_primitive_array::<Float64Type>(n, null_density)) as ArrayRef;
    let fields = Fields::from(vec![
        Field::new("c0", DataType::Int64, true),
        Field::new("c1", DataType::Utf8, true),
        Field::new("c2", DataType::Float64, true),
    ]);
    // Struct-level nulls stay None: `named_struct` never produces a null
    // struct, only null fields — match that shape here.
    Arc::new(StructArray::new(fields, vec![a, b, d], None))
}

/// A `List<Int64>` value column with fixed-size lists of `list_len` elements.
fn create_list_array(n: usize, list_len: usize, null_density: f32) -> ArrayRef {
    let child = Arc::new(create_primitive_array::<Int64Type>(
        n * list_len,
        null_density,
    )) as ArrayRef;
    let offsets = OffsetBuffer::from_lengths(std::iter::repeat_n(list_len, n));
    let field = Arc::new(Field::new_list_field(DataType::Int64, true));
    Arc::new(ListArray::new(field, offsets, child, None))
}

/// Head-to-head for the coalesce-peers rewrite: N independent primitive
/// `first_value` accumulators (the pre-rewrite plan) vs one struct-valued
/// accumulator carrying the same N columns (the post-rewrite plan). Uses the
/// same worst-case ordering as [`update_bench`] so every row forces an
/// ordering comparison in every accumulator.
#[expect(clippy::needless_pass_by_value)]
fn coalesce_comparison_bench(
    c: &mut Criterion,
    name: &str,
    column_values: Vec<ArrayRef>,
    struct_values: ArrayRef,
    ord: ArrayRef,
    num_groups: usize,
) {
    let n = ord.len();
    let group_indices: Vec<usize> = (0..n).map(|i| i % num_groups).collect();
    let worst_ord: ArrayRef = Arc::new(Int64Array::from(vec![i64::MAX; n]));

    // Pre-rewrite: one accumulator per column.
    c.bench_function(&format!("{name} separate x{}", column_values.len()), |b| {
        b.iter_batched(
            || {
                column_values
                    .iter()
                    .map(|values| {
                        let mut acc = prepare_typed_groups_accumulator(
                            true,
                            values.data_type().clone(),
                        );
                        acc.update_batch(
                            &[Arc::clone(values), Arc::clone(&worst_ord)],
                            &group_indices,
                            None,
                            num_groups,
                        )
                        .unwrap();
                        acc
                    })
                    .collect::<Vec<_>>()
            },
            |mut accumulators| {
                for _ in 0..100 {
                    for (acc, values) in accumulators.iter_mut().zip(&column_values) {
                        #[expect(clippy::unit_arg)]
                        black_box(
                            acc.update_batch(
                                &[Arc::clone(values), Arc::clone(&ord)],
                                &group_indices,
                                None,
                                num_groups,
                            )
                            .unwrap(),
                        );
                    }
                }
            },
            BatchSize::SmallInput,
        )
    });

    // Post-rewrite: a single struct-valued accumulator.
    c.bench_function(&format!("{name} coalesced struct"), |b| {
        b.iter_batched(
            || {
                let mut acc = prepare_typed_groups_accumulator(
                    true,
                    struct_values.data_type().clone(),
                );
                acc.update_batch(
                    &[Arc::clone(&struct_values), Arc::clone(&worst_ord)],
                    &group_indices,
                    None,
                    num_groups,
                )
                .unwrap();
                acc
            },
            |mut accumulator| {
                for _ in 0..100 {
                    #[expect(clippy::unit_arg)]
                    black_box(
                        accumulator
                            .update_batch(
                                &[Arc::clone(&struct_values), Arc::clone(&ord)],
                                &group_indices,
                                None,
                                num_groups,
                            )
                            .unwrap(),
                    );
                }
            },
            BatchSize::SmallInput,
        )
    });
}

fn first_last_nested_benchmark(c: &mut Criterion) {
    const N: usize = 65536;
    const NUM_GROUPS: usize = 1024;

    let ord = Arc::new(create_primitive_array::<Int64Type>(N, 0.0)) as ArrayRef;

    for pct in [0, 90] {
        let null_density = (pct as f32) / 100.0;

        let struct_values = create_struct_array(N, null_density);
        update_bench(
            c,
            true,
            &format!("first_value update_bench struct(i64,utf8,f64) nulls={pct}%"),
            struct_values.clone(),
            ord.clone(),
            None,
            NUM_GROUPS,
        );
        evaluate_bench(
            c,
            true,
            EmitTo::All,
            &format!("first_value evaluate_bench struct(i64,utf8,f64) nulls={pct}%, all"),
            struct_values.clone(),
            ord.clone(),
            None,
            NUM_GROUPS,
        );
        merge_bench(
            c,
            true,
            &format!("first_value merge_bench struct(i64,utf8,f64) nulls={pct}%"),
            struct_values.clone(),
            ord.clone(),
            None,
            NUM_GROUPS,
        );

        let list_values = create_list_array(N, 4, null_density);
        update_bench(
            c,
            true,
            &format!("first_value update_bench list<i64>[4] nulls={pct}%"),
            list_values,
            ord.clone(),
            None,
            NUM_GROUPS,
        );
    }

    // Coalesce-peers head-to-head on null-free columns.
    let a = Arc::new(create_primitive_array::<Int64Type>(N, 0.0)) as ArrayRef;
    let b = Arc::new(create_string_array_with_len::<i32>(N, 0.0, 16)) as ArrayRef;
    let d = Arc::new(create_primitive_array::<Float64Type>(N, 0.0)) as ArrayRef;
    let struct_values = create_struct_array(N, 0.0);
    coalesce_comparison_bench(
        c,
        "first_value coalesce_peers(i64,utf8,f64)",
        vec![a, b, d],
        struct_values,
        ord,
        NUM_GROUPS,
    );
}

fn first_last_benchmark(c: &mut Criterion) {
    const N: usize = 65536;
    const NUM_GROUPS: usize = 1024;

    assert_eq!(N % NUM_GROUPS, 0);

    for is_first in [true, false] {
        for pct in [0, 90] {
            let fn_name = if is_first {
                "first_value"
            } else {
                "last_value"
            };

            let null_density = (pct as f32) / 100.0;
            let values = Arc::new(create_primitive_array::<Int64Type>(N, null_density))
                as ArrayRef;
            let ord = Arc::new(create_primitive_array::<Int64Type>(N, null_density))
                as ArrayRef;

            for with_filter in [false, true] {
                let filter = create_boolean_array(N, 0.0, 0.5);
                let opt_filter = if with_filter { Some(&filter) } else { None };

                evaluate_bench(
                    c,
                    is_first,
                    EmitTo::First(2),
                    &format!(
                        "{fn_name} evaluate_bench nulls={pct}%, filter={with_filter}, first(2)"
                    ),
                    values.clone(),
                    ord.clone(),
                    opt_filter,
                    NUM_GROUPS,
                );
                evaluate_bench(
                    c,
                    is_first,
                    EmitTo::All,
                    &format!(
                        "{fn_name} evaluate_bench nulls={pct}%, filter={with_filter}, all"
                    ),
                    values.clone(),
                    ord.clone(),
                    opt_filter,
                    NUM_GROUPS,
                );

                update_bench(
                    c,
                    is_first,
                    &format!("{fn_name} update_bench nulls={pct}%, filter={with_filter}"),
                    values.clone(),
                    ord.clone(),
                    opt_filter,
                    NUM_GROUPS,
                );
                merge_bench(
                    c,
                    is_first,
                    &format!("{fn_name} merge_bench nulls={pct}%, filter={with_filter}"),
                    values.clone(),
                    ord.clone(),
                    opt_filter,
                    NUM_GROUPS,
                );
            }

            for ignore_nulls in [false, true] {
                trivial_update_bench(
                    c,
                    is_first,
                    ignore_nulls,
                    &format!(
                        "{fn_name} trivial_update_bench nulls={pct}%, ignore_nulls={ignore_nulls}"
                    ),
                    values.clone(),
                );
            }
        }
    }
}

criterion_group!(benches, first_last_benchmark, first_last_nested_benchmark);
criterion_main!(benches);
