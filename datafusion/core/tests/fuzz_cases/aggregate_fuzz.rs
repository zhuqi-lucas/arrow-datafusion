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

use std::sync::Arc;

use super::record_batch_generator::get_supported_types_columns;
use crate::fuzz_cases::aggregation_fuzzer::query_builder::QueryBuilder;
use crate::fuzz_cases::aggregation_fuzzer::{
    AggregationFuzzerBuilder, DatasetGeneratorConfig,
};

use arrow::array::{
    types::Int64Type, Array, ArrayRef, AsArray, Int32Array, Int64Array, RecordBatch,
    StringArray,
};
use arrow::compute::concat_batches;
use arrow::datatypes::DataType;
use arrow::util::pretty::pretty_format_batches;
use arrow_schema::{Field, Schema, SchemaRef};
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::datasource::source::DataSourceExec;
use datafusion::datasource::MemTable;
use datafusion::prelude::{DataFrame, SessionConfig, SessionContext};
use datafusion_common::tree_node::{TreeNode, TreeNodeRecursion, TreeNodeVisitor};
use datafusion_common::{HashMap, Result};
use datafusion_common_runtime::JoinSet;
use datafusion_functions_aggregate::sum::sum_udaf;
use datafusion_physical_expr::expressions::{col, lit, Column};
use datafusion_physical_expr::PhysicalSortExpr;
use datafusion_physical_plan::InputOrderMode;
use test_utils::{add_empty_batches, StringBatchGenerator};

use datafusion_execution::memory_pool::FairSpillPool;
use datafusion_execution::runtime_env::RuntimeEnvBuilder;
use datafusion_execution::TaskContext;
use datafusion_physical_expr::aggregate::AggregateExprBuilder;
use datafusion_physical_plan::aggregates::{
    AggregateExec, AggregateMode, PhysicalGroupBy,
};
use datafusion_physical_plan::metrics::MetricValue;
use datafusion_physical_plan::{collect, displayable, ExecutionPlan};
use rand::rngs::StdRng;
use rand::{random, rng, Rng, SeedableRng};

// ========================================================================
//  The new aggregation fuzz tests based on [`AggregationFuzzer`]
// ========================================================================
//
// Notes on tests:
//
// Since the supported types differ for each aggregation function, the tests
// below are structured so they enumerate each different aggregate function.
//
// The test framework handles varying combinations of arguments (data types),
// sortedness, and grouping parameters
//
// TODO: Test floating point values (where output needs to be compared with some
// acceptable range due to floating point rounding)
//
// TODO: test other aggregate functions
// - AVG (unstable given the wide range of inputs)
#[tokio::test(flavor = "multi_thread")]
async fn test_min() {
    let data_gen_config = baseline_config();

    // Queries like SELECT min(a) FROM fuzz_table GROUP BY b
    let query_builder = QueryBuilder::new()
        .with_table_name("fuzz_table")
        .with_aggregate_function("min")
        // min works on all column types
        .with_aggregate_arguments(data_gen_config.all_columns())
        .with_dataset_sort_keys(data_gen_config.sort_keys_set.clone())
        .set_group_by_columns(data_gen_config.all_columns());

    AggregationFuzzerBuilder::from(data_gen_config)
        .add_query_builder(query_builder)
        .build()
        .run()
        .await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_first_val() {
    let mut data_gen_config: DatasetGeneratorConfig = baseline_config();

    for i in 0..data_gen_config.columns.len() {
        if data_gen_config.columns[i].get_max_num_distinct().is_none() {
            data_gen_config.columns[i] = data_gen_config.columns[i]
                .clone()
                // Minimize the chance of identical values in the order by columns to make the test more stable
                .with_max_num_distinct(usize::MAX);
        }
    }

    let query_builder = QueryBuilder::new()
        .with_table_name("fuzz_table")
        .with_aggregate_function("first_value")
        .with_aggregate_arguments(data_gen_config.all_columns())
        .with_dataset_sort_keys(data_gen_config.sort_keys_set.clone())
        .set_group_by_columns(data_gen_config.all_columns());

    AggregationFuzzerBuilder::from(data_gen_config)
        .add_query_builder(query_builder)
        .build()
        .run()
        .await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_last_val() {
    let mut data_gen_config = baseline_config();

    for i in 0..data_gen_config.columns.len() {
        if data_gen_config.columns[i].get_max_num_distinct().is_none() {
            data_gen_config.columns[i] = data_gen_config.columns[i]
                .clone()
                // Minimize the chance of identical values in the order by columns to make the test more stable
                .with_max_num_distinct(usize::MAX);
        }
    }

    let query_builder = QueryBuilder::new()
        .with_table_name("fuzz_table")
        .with_aggregate_function("last_value")
        .with_aggregate_arguments(data_gen_config.all_columns())
        .with_dataset_sort_keys(data_gen_config.sort_keys_set.clone())
        .set_group_by_columns(data_gen_config.all_columns());

    AggregationFuzzerBuilder::from(data_gen_config)
        .add_query_builder(query_builder)
        .build()
        .run()
        .await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_max() {
    let data_gen_config = baseline_config();

    // Queries like SELECT max(a) FROM fuzz_table GROUP BY b
    let query_builder = QueryBuilder::new()
        .with_table_name("fuzz_table")
        .with_aggregate_function("max")
        // max works on all column types
        .with_aggregate_arguments(data_gen_config.all_columns())
        .with_dataset_sort_keys(data_gen_config.sort_keys_set.clone())
        .set_group_by_columns(data_gen_config.all_columns());

    AggregationFuzzerBuilder::from(data_gen_config)
        .add_query_builder(query_builder)
        .build()
        .run()
        .await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_sum() {
    let data_gen_config = baseline_config();

    // Queries like SELECT sum(a), sum(distinct) FROM fuzz_table GROUP BY b
    let query_builder = QueryBuilder::new()
        .with_table_name("fuzz_table")
        .with_aggregate_function("sum")
        .with_distinct_aggregate_function("sum")
        // sum only works on numeric columns
        .with_aggregate_arguments(data_gen_config.numeric_columns())
        .with_dataset_sort_keys(data_gen_config.sort_keys_set.clone())
        .set_group_by_columns(data_gen_config.all_columns());

    AggregationFuzzerBuilder::from(data_gen_config)
        .add_query_builder(query_builder)
        .build()
        .run()
        .await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_count() {
    let data_gen_config = baseline_config();

    // Queries like SELECT count(a), count(distinct) FROM fuzz_table GROUP BY b
    let query_builder = QueryBuilder::new()
        .with_table_name("fuzz_table")
        .with_aggregate_function("count")
        .with_distinct_aggregate_function("count")
        // count work for all arguments
        .with_aggregate_arguments(data_gen_config.all_columns())
        .with_dataset_sort_keys(data_gen_config.sort_keys_set.clone())
        .set_group_by_columns(data_gen_config.all_columns());

    AggregationFuzzerBuilder::from(data_gen_config)
        .add_query_builder(query_builder)
        .build()
        .run()
        .await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_median() {
    let data_gen_config = baseline_config();

    // Queries like SELECT median(a), median(distinct) FROM fuzz_table GROUP BY b
    let query_builder = QueryBuilder::new()
        .with_table_name("fuzz_table")
        .with_aggregate_function("median")
        .with_distinct_aggregate_function("median")
        // median only works on numeric columns
        .with_aggregate_arguments(data_gen_config.numeric_columns())
        .with_dataset_sort_keys(data_gen_config.sort_keys_set.clone())
        .set_group_by_columns(data_gen_config.all_columns());

    AggregationFuzzerBuilder::from(data_gen_config)
        .add_query_builder(query_builder)
        .build()
        .run()
        .await;
}

/// Return a standard set of columns for testing data generation
///
/// Includes numeric and string types
///
/// Does not include:
/// 1. Floating point numbers
/// 1. structured types
fn baseline_config() -> DatasetGeneratorConfig {
    let mut rng = rng();
    let columns = get_supported_types_columns(rng.random());

    let min_num_rows = 512;
    let max_num_rows = 1024;

    DatasetGeneratorConfig {
        columns,
        rows_num_range: (min_num_rows, max_num_rows),
        sort_keys_set: vec![
            // low cardinality to try and get many repeated runs
            vec![String::from("u8_low")],
            vec![String::from("utf8_low"), String::from("u8_low")],
            vec![String::from("dictionary_utf8_low")],
            vec![
                String::from("dictionary_utf8_low"),
                String::from("utf8_low"),
                String::from("u8_low"),
            ],
        ],
    }
}

// ========================================================================
//  The old aggregation fuzz tests
// ========================================================================

/// Tracks if this stream is generating input or output
/// Tests that streaming aggregate and batch (non streaming) aggregate produce
/// same results
#[tokio::test(flavor = "multi_thread")]
async fn streaming_aggregate_test() {
    let test_cases = [
        vec!["a"],
        vec!["b", "a"],
        vec!["c", "a"],
        vec!["c", "b", "a"],
        vec!["d", "a"],
        vec!["d", "b", "a"],
        vec!["d", "c", "a"],
        vec!["d", "c", "b", "a"],
    ];
    let n = 10;
    let distincts = vec![10, 20];
    for distinct in distincts {
        let mut join_set = JoinSet::new();
        for i in 0..n {
            let test_idx = i % test_cases.len();
            let group_by_columns = test_cases[test_idx].clone();
            join_set.spawn(run_aggregate_test(
                make_staggered_batches::<true>(1000, distinct, i as u64),
                group_by_columns,
            ));
        }
        while let Some(join_handle) = join_set.join_next().await {
            // propagate errors
            join_handle.unwrap();
        }
    }
}

/// Perform batch and streaming aggregation with same input
/// and verify outputs of `AggregateExec` with pipeline breaking stream `GroupedHashAggregateStream`
/// and non-pipeline breaking stream `BoundedAggregateStream` produces same result.
async fn run_aggregate_test(input1: Vec<RecordBatch>, group_by_columns: Vec<&str>) {
    let schema = input1[0].schema();
    let session_config = SessionConfig::new().with_batch_size(50);
    let ctx = SessionContext::new_with_config(session_config);
    let sort_keys = ["a", "b", "c"].map(|ordering_col| {
        PhysicalSortExpr::new_default(col(ordering_col, &schema).unwrap())
    });

    let concat_input_record = concat_batches(&schema, &input1).unwrap();

    let usual_source = MemorySourceConfig::try_new_exec(
        &[vec![concat_input_record]],
        schema.clone(),
        None,
    )
    .unwrap();

    let running_source = DataSourceExec::from_data_source(
        MemorySourceConfig::try_new(&[input1.clone()], schema.clone(), None)
            .unwrap()
            .try_with_sort_information(vec![sort_keys.into()])
            .unwrap(),
    );

    let aggregate_expr =
        vec![
            AggregateExprBuilder::new(sum_udaf(), vec![col("d", &schema).unwrap()])
                .schema(Arc::clone(&schema))
                .alias("sum1")
                .build()
                .map(Arc::new)
                .unwrap(),
        ];
    let expr = group_by_columns
        .iter()
        .map(|elem| (col(elem, &schema).unwrap(), elem.to_string()))
        .collect::<Vec<_>>();
    let group_by = PhysicalGroupBy::new_single(expr);

    let aggregate_exec_running = Arc::new(
        AggregateExec::try_new(
            AggregateMode::Partial,
            group_by.clone(),
            aggregate_expr.clone(),
            vec![None],
            running_source,
            schema.clone(),
        )
        .unwrap(),
    ) as Arc<dyn ExecutionPlan>;

    let aggregate_exec_usual = Arc::new(
        AggregateExec::try_new(
            AggregateMode::Partial,
            group_by.clone(),
            aggregate_expr.clone(),
            vec![None],
            usual_source,
            schema.clone(),
        )
        .unwrap(),
    ) as Arc<dyn ExecutionPlan>;

    let task_ctx = ctx.task_ctx();
    let collected_usual = collect(aggregate_exec_usual.clone(), task_ctx.clone())
        .await
        .unwrap();

    let collected_running = collect(aggregate_exec_running.clone(), task_ctx.clone())
        .await
        .unwrap();
    assert!(collected_running.len() > 2);
    // Running should produce more chunk than the usual AggregateExec.
    // Otherwise it means that we cannot generate result in running mode.
    assert!(collected_running.len() > collected_usual.len());
    // compare
    let usual_formatted = pretty_format_batches(&collected_usual).unwrap().to_string();
    let running_formatted = pretty_format_batches(&collected_running)
        .unwrap()
        .to_string();

    let mut usual_formatted_sorted: Vec<&str> = usual_formatted.trim().lines().collect();
    usual_formatted_sorted.sort_unstable();

    let mut running_formatted_sorted: Vec<&str> =
        running_formatted.trim().lines().collect();
    running_formatted_sorted.sort_unstable();
    for (i, (usual_line, running_line)) in usual_formatted_sorted
        .iter()
        .zip(&running_formatted_sorted)
        .enumerate()
    {
        assert_eq!(
            (i, usual_line),
            (i, running_line),
            "Inconsistent result\n\n\
             Aggregate_expr: {aggregate_expr:?}\n\
             group_by: {group_by:?}\n\
             Left Plan:\n{}\n\
             Right Plan:\n{}\n\
             schema:\n{schema}\n\
             Left Ouptut:\n{}\n\
             Right Output:\n{}\n\
             input:\n{}\n\
             ",
            displayable(aggregate_exec_usual.as_ref()).indent(false),
            displayable(aggregate_exec_running.as_ref()).indent(false),
            usual_formatted,
            running_formatted,
            pretty_format_batches(&input1).unwrap(),
        );
    }
}

/// Return randomly sized record batches with:
/// three sorted int64 columns 'a', 'b', 'c' ranged from 0..'n_distinct' as columns
/// one random int64 column 'd' as other columns
pub(crate) fn make_staggered_batches<const STREAM: bool>(
    len: usize,
    n_distinct: usize,
    random_seed: u64,
) -> Vec<RecordBatch> {
    // use a random number generator to pick a random sized output
    let mut rng = StdRng::seed_from_u64(random_seed);
    let mut input123: Vec<(i64, i64, i64)> = vec![(0, 0, 0); len];
    let mut input4: Vec<i64> = vec![0; len];
    input123.iter_mut().for_each(|v| {
        *v = (
            rng.random_range(0..n_distinct) as i64,
            rng.random_range(0..n_distinct) as i64,
            rng.random_range(0..n_distinct) as i64,
        )
    });
    input4.iter_mut().for_each(|v| {
        *v = rng.random_range(0..n_distinct) as i64;
    });
    input123.sort();
    let input1 = Int64Array::from_iter_values(input123.clone().into_iter().map(|k| k.0));
    let input2 = Int64Array::from_iter_values(input123.clone().into_iter().map(|k| k.1));
    let input3 = Int64Array::from_iter_values(input123.clone().into_iter().map(|k| k.2));
    let input4 = Int64Array::from_iter_values(input4);

    // split into several record batches
    let mut remainder = RecordBatch::try_from_iter(vec![
        ("a", Arc::new(input1) as ArrayRef),
        ("b", Arc::new(input2) as ArrayRef),
        ("c", Arc::new(input3) as ArrayRef),
        ("d", Arc::new(input4) as ArrayRef),
    ])
    .unwrap();

    let mut batches = vec![];
    if STREAM {
        while remainder.num_rows() > 0 {
            let batch_size = rng.random_range(0..50);
            if remainder.num_rows() < batch_size {
                break;
            }
            batches.push(remainder.slice(0, batch_size));
            remainder = remainder.slice(batch_size, remainder.num_rows() - batch_size);
        }
    } else {
        while remainder.num_rows() > 0 {
            let batch_size = rng.random_range(0..remainder.num_rows() + 1);
            batches.push(remainder.slice(0, batch_size));
            remainder = remainder.slice(batch_size, remainder.num_rows() - batch_size);
        }
    }
    add_empty_batches(batches, &mut rng)
}

/// Test group by with string/large string columns
#[tokio::test(flavor = "multi_thread")]
async fn group_by_strings() {
    let mut join_set = JoinSet::new();
    for large in [true, false] {
        for sorted in [true, false] {
            for generator in StringBatchGenerator::interesting_cases() {
                join_set.spawn(group_by_string_test(generator, sorted, large));
            }
        }
    }
    while let Some(join_handle) = join_set.join_next().await {
        // propagate errors
        join_handle.unwrap();
    }
}

/// Run GROUP BY <x> using SQL and ensure the results are correct
///
/// If sorted is true, the input batches will be sorted by the group by column
/// to test the streaming group by case
///
/// if large is true, the input batches will be LargeStringArray
async fn group_by_string_test(
    mut generator: StringBatchGenerator,
    sorted: bool,
    large: bool,
) {
    let column_name = "a";
    let input = if sorted {
        generator.make_sorted_input_batches(large)
    } else {
        generator.make_input_batches()
    };

    let expected = compute_counts(&input, column_name);

    let schema = input[0].schema();
    let session_config = SessionConfig::new()
        .with_batch_size(50)
        .with_repartition_file_scans(false);
    let ctx = SessionContext::new_with_config(session_config);

    let provider = MemTable::try_new(schema.clone(), vec![input]).unwrap();
    let provider = if sorted {
        let sort_expr = datafusion::prelude::col("a").sort(true, true);
        provider.with_sort_order(vec![vec![sort_expr]])
    } else {
        provider
    };

    ctx.register_table("t", Arc::new(provider)).unwrap();

    let df = ctx
        .sql("SELECT a, COUNT(*) FROM t GROUP BY a")
        .await
        .unwrap();
    verify_ordered_aggregate(&df, sorted).await;
    let results = df.collect().await.unwrap();

    // verify that the results are correct
    let actual = extract_result_counts(results);
    assert_eq!(expected, actual);
}

async fn verify_ordered_aggregate(frame: &DataFrame, expected_sort: bool) {
    struct Visitor {
        expected_sort: bool,
    }
    let mut visitor = Visitor { expected_sort };

    impl<'n> TreeNodeVisitor<'n> for Visitor {
        type Node = Arc<dyn ExecutionPlan>;

        fn f_down(&mut self, node: &'n Self::Node) -> Result<TreeNodeRecursion> {
            if let Some(exec) = node.as_any().downcast_ref::<AggregateExec>() {
                if self.expected_sort {
                    assert!(matches!(
                        exec.input_order_mode(),
                        InputOrderMode::PartiallySorted(_) | InputOrderMode::Sorted
                    ));
                } else {
                    assert!(matches!(exec.input_order_mode(), InputOrderMode::Linear));
                }
            }
            Ok(TreeNodeRecursion::Continue)
        }
    }

    let plan = frame.clone().create_physical_plan().await.unwrap();
    plan.visit(&mut visitor).unwrap();
}

/// Compute the count of each distinct value in the specified column
///
/// ```text
/// +---------------+---------------+
/// | a             | b             |
/// +---------------+---------------+
/// | 𭏷񬝜󓴻𼇪󄶛𑩁򽵐󦊟    | 󺚤𘱦𫎛񐕿        |
/// | 󂌿󶴬񰶨񺹭𿑵󖺉       | 񥼧􋽐󮋋󑤐𬿪𜋃       |
/// ```
fn compute_counts(batches: &[RecordBatch], col: &str) -> HashMap<Option<String>, i64> {
    let mut output = HashMap::new();
    for arr in batches
        .iter()
        .map(|batch| batch.column_by_name(col).unwrap())
    {
        for value in to_str_vec(arr) {
            output.entry(value).and_modify(|e| *e += 1).or_insert(1);
        }
    }
    output
}

fn to_str_vec(array: &ArrayRef) -> Vec<Option<String>> {
    match array.data_type() {
        DataType::Utf8 => array
            .as_string::<i32>()
            .iter()
            .map(|x| x.map(|x| x.to_string()))
            .collect(),
        DataType::LargeUtf8 => array
            .as_string::<i64>()
            .iter()
            .map(|x| x.map(|x| x.to_string()))
            .collect(),
        _ => panic!("unexpected type"),
    }
}

/// extracts the value of the first column and the count of the second column
/// ```text
/// +----------------+----------+
/// | a              | COUNT(*) |
/// +----------------+----------+
/// | 񩢰񴠍             | 8        |
/// | 󇿺򷜄򩨝񜖫𑟑񣶏󣥽𹕉      | 11       |
/// ```
fn extract_result_counts(results: Vec<RecordBatch>) -> HashMap<Option<String>, i64> {
    let group_arrays = results.iter().map(|batch| batch.column(0));

    let count_arrays = results
        .iter()
        .map(|batch| batch.column(1).as_primitive::<Int64Type>());

    let mut output = HashMap::new();
    for (group_arr, count_arr) in group_arrays.zip(count_arrays) {
        assert_eq!(group_arr.len(), count_arr.len());
        let group_values = to_str_vec(group_arr);
        for (group, count) in group_values.into_iter().zip(count_arr.iter()) {
            assert!(output.get(&group).is_none());
            let count = count.unwrap(); // counts can never be null
            output.insert(group, count);
        }
    }
    output
}

pub(crate) fn assert_spill_count_metric(
    expect_spill: bool,
    plan_that_spills: Arc<dyn ExecutionPlan>,
) -> usize {
    if let Some(metrics_set) = plan_that_spills.metrics() {
        let mut spill_count = 0;

        // Inspect metrics for SpillCount
        for metric in metrics_set.iter() {
            if let MetricValue::SpillCount(count) = metric.value() {
                spill_count = count.value();
                break;
            }
        }

        if expect_spill && spill_count == 0 {
            panic!("Expected spill but SpillCount metric not found or SpillCount was 0.");
        } else if !expect_spill && spill_count > 0 {
            panic!("Expected no spill but found SpillCount metric with value greater than 0.");
        }

        spill_count
    } else {
        panic!("No metrics returned from the operator; cannot verify spilling.");
    }
}

// Fix for https://github.com/apache/datafusion/issues/15530
#[tokio::test]
async fn test_single_mode_aggregate_single_mode_aggregate_with_spill() -> Result<()> {
    let scan_schema = Arc::new(Schema::new(vec![
        Field::new("col_0", DataType::Int64, true),
        Field::new("col_1", DataType::Utf8, true),
        Field::new("col_2", DataType::Utf8, true),
        Field::new("col_3", DataType::Utf8, true),
        Field::new("col_4", DataType::Utf8, true),
        Field::new("col_5", DataType::Int32, true),
        Field::new("col_6", DataType::Utf8, true),
        Field::new("col_7", DataType::Utf8, true),
        Field::new("col_8", DataType::Utf8, true),
    ]));

    let group_by = PhysicalGroupBy::new_single(vec![
        (Arc::new(Column::new("col_1", 1)), "col_1".to_string()),
        (Arc::new(Column::new("col_7", 7)), "col_7".to_string()),
        (Arc::new(Column::new("col_0", 0)), "col_0".to_string()),
        (Arc::new(Column::new("col_8", 8)), "col_8".to_string()),
    ]);

    fn generate_int64_array() -> ArrayRef {
        Arc::new(Int64Array::from_iter_values(
            (0..1024).map(|_| random::<i64>()),
        ))
    }
    fn generate_int32_array() -> ArrayRef {
        Arc::new(Int32Array::from_iter_values(
            (0..1024).map(|_| random::<i32>()),
        ))
    }

    fn generate_string_array() -> ArrayRef {
        Arc::new(StringArray::from(
            (0..1024)
                .map(|_| -> String {
                    rng()
                        .sample_iter::<char, _>(rand::distr::StandardUniform)
                        .take(5)
                        .collect()
                })
                .collect::<Vec<_>>(),
        ))
    }

    fn generate_record_batch(schema: &SchemaRef) -> Result<RecordBatch> {
        RecordBatch::try_new(
            Arc::clone(schema),
            vec![
                generate_int64_array(),
                generate_string_array(),
                generate_string_array(),
                generate_string_array(),
                generate_string_array(),
                generate_int32_array(),
                generate_string_array(),
                generate_string_array(),
                generate_string_array(),
            ],
        )
        .map_err(|err| err.into())
    }

    let aggregate_expressions = vec![Arc::new(
        AggregateExprBuilder::new(sum_udaf(), vec![lit(1i64)])
            .schema(Arc::clone(&scan_schema))
            .alias("SUM(1i64)")
            .build()?,
    )];

    let batches = (0..5)
        .map(|_| generate_record_batch(&scan_schema))
        .collect::<Result<Vec<_>>>()?;

    let plan: Arc<dyn ExecutionPlan> =
        MemorySourceConfig::try_new_exec(&[batches], Arc::clone(&scan_schema), None)
            .unwrap();

    let single_aggregate = Arc::new(AggregateExec::try_new(
        AggregateMode::Single,
        group_by,
        aggregate_expressions.clone(),
        vec![None; aggregate_expressions.len()],
        plan,
        Arc::clone(&scan_schema),
    )?);

    let memory_pool = Arc::new(FairSpillPool::new(250000));
    let task_ctx = Arc::new(
        TaskContext::default()
            .with_session_config(SessionConfig::new().with_batch_size(248))
            .with_runtime(Arc::new(
                RuntimeEnvBuilder::new()
                    .with_memory_pool(memory_pool)
                    .build()?,
            )),
    );

    datafusion_physical_plan::common::collect(
        single_aggregate.execute(0, Arc::clone(&task_ctx))?,
    )
    .await?;

    assert_spill_count_metric(true, single_aggregate);

    Ok(())
}
