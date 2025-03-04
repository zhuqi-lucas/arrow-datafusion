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

use std::fmt::{Display, Formatter};
use std::pin::Pin;
use std::str::FromStr;

use crate::print_format::PrintFormat;

use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use datafusion::common::instant::Instant;
use datafusion::error::Result;
use datafusion::physical_plan::RecordBatchStream;

use datafusion::execution::SendableRecordBatchStream;
use futures::StreamExt;

#[derive(Debug, Clone, PartialEq, Copy)]
pub enum MaxRows {
    /// show all rows in the output
    Unlimited,
    /// Only show n rows
    Limited(usize),
}

impl FromStr for MaxRows {
    type Err = String;

    fn from_str(maxrows: &str) -> Result<Self, Self::Err> {
        if maxrows.to_lowercase() == "inf"
            || maxrows.to_lowercase() == "infinite"
            || maxrows.to_lowercase() == "none"
        {
            Ok(Self::Unlimited)
        } else {
            match maxrows.parse::<usize>() {
                Ok(nrows)  => Ok(Self::Limited(nrows)),
                _ => Err(format!("Invalid maxrows {}. Valid inputs are natural numbers or \'none\', \'inf\', or \'infinite\' for no limit.", maxrows)),
            }
        }
    }
}

impl Display for MaxRows {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unlimited => write!(f, "unlimited"),
            Self::Limited(max_rows) => write!(f, "at most {max_rows}"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PrintOptions {
    pub format: PrintFormat,
    pub quiet: bool,
    pub maxrows: MaxRows,
    pub color: bool,
}

impl PrintOptions {
    /// Prints the stream to stdout using the table format.
    ///
    /// This function processes a stream of record batches, formats them into a table,
    /// and writes the output to the provided writer. It handles scenarios where the
    /// number of rows exceeds the specified `max_rows` and ensures that the table
    /// header and borders are printed appropriately.
    ///
    /// # Arguments
    ///
    /// * `schema` - The schema reference of the record batches.
    /// * `stream` - The mutable stream of record batches to be printed.
    /// * `max_rows` - The maximum number of rows to print.
    /// * `writer` - The mutable writer to which the formatted table will be written.
    /// * `now` - The instant representing the start time of the operation.
    ///
    /// # Returns
    ///
    /// A `Result` indicating the success or failure of the operation.
    pub async fn print_streaming_table_batch<W: std::io::Write>(
        &self,
        schema: SchemaRef,
        stream: &mut SendableRecordBatchStream,
        max_rows: usize,
        writer: &mut W,
        now: Instant,
    ) -> Result<()> {
        let preview_limit: usize = 1000;
        let mut preview_batches: Vec<RecordBatch> = vec![];
        let mut preview_row_count = 0_usize;
        let mut total_count = 0_usize;
        let mut precomputed_widths: Option<Vec<usize>> = None;
        let mut header_printed = false;
        let mut max_rows_reached = false;

        // Iterate over each batch in the stream
        while let Some(batch) = stream.next().await {
            let batch = batch?;
            let batch_rows = batch.num_rows();

            // Check if the maximum number of rows to print has been reached
            if !max_rows_reached && total_count < max_rows {
                if total_count + batch_rows > max_rows {
                    // If adding this batch exceeds max_rows, slice the batch to fit
                    let needed = max_rows - total_count;
                    let batch_to_print = batch.slice(0, needed);
                    self.format.process_table_batch(
                        &batch_to_print,
                        schema.clone(),
                        &mut preview_batches,
                        &mut preview_row_count,
                        preview_limit,
                        &mut precomputed_widths,
                        &mut header_printed,
                        writer,
                    )?;
                    if precomputed_widths.is_none() {
                        // Compute column widths if not already done
                        let widths = self
                            .format
                            .compute_column_widths(&preview_batches, schema.clone())?;
                        precomputed_widths = Some(widths.clone());
                        if !header_printed {
                            // Print the table header
                            self.format.print_header(&schema, &widths, writer)?;
                        }
                        // Print each preview batch with the computed widths
                        for preview_batch in preview_batches.drain(..) {
                            self.format.print_batch_with_widths(
                                &preview_batch,
                                &widths,
                                writer,
                            )?;
                        }
                    }
                    if let Some(ref widths) = precomputed_widths {
                        // Print dotted lines and bottom border to indicate truncation
                        for _ in 0..3 {
                            self.format.print_dotted_line(widths, writer)?;
                        }
                        self.format.print_bottom_border(widths, writer)?;
                    }
                    max_rows_reached = true;
                } else {
                    // Process the entire batch if it doesn't exceed max_rows
                    self.format.process_table_batch(
                        &batch,
                        schema.clone(),
                        &mut preview_batches,
                        &mut preview_row_count,
                        preview_limit,
                        &mut precomputed_widths,
                        &mut header_printed,
                        writer,
                    )?;
                }
            }

            total_count += batch_rows;
        }

        // Handle the case where the maximum number of rows was not reached
        if !max_rows_reached {
            if precomputed_widths.is_none() {
                // Compute column widths if not already done
                if preview_batches.is_empty() {
                    // If no rows are present, set column widths based on header names
                    let widths = schema
                        .fields()
                        .iter()
                        .map(|f| f.name().len())
                        .collect::<Vec<usize>>();
                    precomputed_widths = Some(widths);
                } else {
                    let widths = self
                        .format
                        .compute_column_widths(&preview_batches, schema.clone())?;
                    precomputed_widths = Some(widths);
                    if !header_printed {
                        // Print the table header
                        self.format.print_header(
                            &schema,
                            precomputed_widths.as_ref().unwrap(),
                            writer,
                        )?;
                        header_printed = true;
                    }
                    // Print each preview batch with the computed widths
                    // Here we make sure the preview_batches are printed even if the total count is less than max_rows
                    for preview_batch in preview_batches.drain(..) {
                        self.format.print_batch_with_widths(
                            &preview_batch,
                            precomputed_widths.as_ref().unwrap(),
                            writer,
                        )?;
                    }
                }
                if !header_printed {
                    // Print the table header if not already printed
                    self.format.print_header(
                        &schema,
                        precomputed_widths.as_ref().unwrap(),
                        writer,
                    )?;
                }
            }
            if let Some(ref widths) = precomputed_widths {
                // Print the bottom border of the table
                self.format.print_bottom_border(widths, writer)?;
            }
        }

        // Get and print the formatted execution details
        let formatted_exec_details =
            self.get_execution_details_formatted(total_count, now);

        if !self.quiet {
            writeln!(writer, "{formatted_exec_details}")?;
        }

        Ok(())
    }

    /// Print the stream to stdout using the format which is not table format
    pub async fn print_no_table_streaming_batch<W: std::io::Write>(
        &self,
        stream: &mut SendableRecordBatchStream,
        writer: &mut W,
        now: Instant,
    ) -> Result<()> {
        let max_count = match self.maxrows {
            MaxRows::Unlimited => usize::MAX,
            MaxRows::Limited(n) => n,
        };

        let mut row_count = 0_usize;
        let mut with_header = true;
        let mut max_rows_reached = false;

        while let Some(maybe_batch) = stream.next().await {
            let batch = maybe_batch?;
            let curr_batch_rows = batch.num_rows();
            if !max_rows_reached && row_count < max_count {
                if row_count + curr_batch_rows > max_count {
                    let needed = max_count - row_count;
                    let batch_to_print = batch.slice(0, needed);
                    self.format.print_no_table_batches(
                        writer,
                        batch.schema(),
                        &[batch_to_print],
                        with_header,
                    )?;
                    max_rows_reached = true;
                } else {
                    self.format.print_no_table_batches(
                        writer,
                        batch.schema(),
                        &[batch],
                        with_header,
                    )?;
                }
            }
            row_count += curr_batch_rows;
            with_header = false;
        }

        let formatted_exec_details = self.get_execution_details_formatted(row_count, now);

        if !self.quiet {
            writeln!(writer, "{formatted_exec_details}")?;
        }

        Ok(())
    }

    /// Print the stream to stdout using the specified format
    /// There are two modes of operation:
    /// 1. If the format is table, the stream is processed in batches and previewed to determine the column widths
    ///    before printing the full result set. And after we have the column widths, we print batch by batch with the correct widths.
    ///
    /// 2. If the format is not table, the stream is processed batch by batch and printed immediately.
    ///
    /// The query_start_time is used to calculate the elapsed time for the query.
    /// The schema is used to print the header.
    pub async fn print_stream(
        &self,
        schema: SchemaRef,
        mut stream: Pin<Box<dyn RecordBatchStream + Send>>,
        query_start_time: Instant,
    ) -> Result<()> {
        let max_count = match self.maxrows {
            MaxRows::Unlimited => usize::MAX,
            MaxRows::Limited(n) => n,
        };

        let stdout = std::io::stdout();
        let mut writer = stdout.lock();

        if self.format == PrintFormat::Table {
            self.print_streaming_table_batch(
                schema,
                &mut stream,
                max_count,
                &mut writer,
                query_start_time,
            )
            .await?;
        } else {
            self.print_no_table_streaming_batch(
                &mut stream,
                &mut writer,
                query_start_time,
            )
            .await?;
        }

        Ok(())
    }

    // Returns the query execution details formatted
    pub fn get_execution_details_formatted(
        &self,
        row_count: usize,
        query_start_time: Instant,
    ) -> String {
        let nrows_shown_msg = match self.maxrows {
            MaxRows::Limited(nrows) if nrows < row_count => {
                format!("(First {nrows} displayed. Use --maxrows to adjust)")
            }
            _ => String::new(),
        };

        format!(
            "{} row(s) fetched. {}\nElapsed {:.3} seconds.\n",
            row_count,
            nrows_shown_msg,
            query_start_time.elapsed().as_secs_f64()
        )
    }
}
