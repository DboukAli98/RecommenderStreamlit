import re
import pyodbc
import sqlalchemy as sa
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os
from dotenv import load_dotenv
from deprecated import deprecated


# loading dotenv
load_dotenv()


# regex to prevent SQL Injection for "TableName" or "Schema.TableName" validations
_VALID_TBL = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?$")


class SQLServerService:
    """

    SQL Server DAL class fpor retrieval and insertion of data

    """

    def __init__(
        self,
        driver="{ODBC Driver 18 for SQL Server}",
    ):
        self.server = os.getenv("SQL_SERVER_SERVER")
        self.database = os.getenv("SQL_SERVER_DATABASE")
        self.user = os.getenv("SQL_SERVER_USER")
        self.password = os.getenv("SQL_SERVER_PASSWORD")

        if self.user and self.password:
            self.conn_str = (
                f"DRIVER={driver};"
                f"SERVER={self.server};"
                f"DATABASE={self.database};"
                f"UID={self.user};"
                f"PWD={self.password};"
                f"Encrypt=yes;"
                f"TrustServerCertificate=yes;"
            )
        else:
            self.conn_str = (
                f"DRIVER={driver};"
                f"SERVER={self.server};"
                f"DATABASE={self.database};"
                f"Trusted_Connection=yes;"
                f"TrustServerCertificate=yes;"
            )

    # def query_to_df(self, query: str, chunksize: int = 50000) -> pd.DataFrame:
    #     """
    #     Execute a SELECT query from the database and return the results as a Pandas DataFrame.
    #     Uses chunked loading for better performance on large datasets.

    #     Parameters:
    #     - query (str): SQL SELECT query to execute.
    #     - chunksize (int): Number of rows per chunk to fetch. Default is 50,000.

    #     Returns:
    #     - pd.DataFrame: Combined result of all fetched chunks.
    #     """
    #     try:
    #         with pyodbc.connect(self.conn_str) as conn:
    #             chunks = []
    #             for i, chunk in enumerate(
    #                 pd.read_sql(query, conn, chunksize=chunksize)
    #             ):
    #                 print(f"Fetched chunk {i + 1} with {len(chunk)} rows")
    #                 chunks.append(chunk)

    #             df = pd.concat(chunks, ignore_index=True)
    #             logging.info(
    #                 "Query executed successfully with chunksize=%d: %s",
    #                 chunksize,
    #                 query,
    #             )
    #             return df
    #     except Exception as e:
    #         logging.error("Failed to execute query: %s", e)
    #         raise

    def query_to_df(self, query: str, chunksize: int = 50000) -> pd.DataFrame:
        """
        Execute a SELECT query from the database and return the results as a Pandas DataFrame.
        Uses chunked loading for better performance on large datasets.

        Parameters:
        - query (str): SQL SELECT query to execute.
        - chunksize (int): Number of rows per chunk to fetch. Default is 50,000.

        Returns:
        - pd.DataFrame: Combined result of all fetched chunks.
        """
        try:
            # Creating SQLAlchemy engine
            engine = sa.create_engine(f"mssql+pyodbc:///?odbc_connect={self.conn_str}")

            # Executing query in chunks
            chunks = []
            with engine.connect() as connection:
                for i, chunk in enumerate(
                    pd.read_sql(query, connection, chunksize=chunksize)
                ):
                    print(f"Fetched chunk {i + 1} with {len(chunk)} rows")
                    chunks.append(chunk)

            # Combining chunks into a single DataFrame
            df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

            logging.info(
                "Query executed successfully with chunksize=%d: %s",
                chunksize,
                query,
            )
            return df
        except Exception as e:
            logging.error("Failed to execute query: %s", e)
            raise

    def delete_all_rows(self, table_name: str):
        """
        Delete all rows from the specified table.

        Parameters:
        - table_name: The SQL Server table to clear

        Returns:
        - Number of rows deleted
        """
        try:
            with pyodbc.connect(self.conn_str) as conn:
                cursor = conn.cursor()

                delete_sql = f"DELETE FROM {table_name}"
                cursor.execute(delete_sql)

                rows_deleted = cursor.rowcount

                conn.commit()

                logging.info("Deleted %d rows from %s", rows_deleted, table_name)
                return rows_deleted
        except Exception as e:
            logging.error("Failed to delete rows from %s: %s", table_name, e)
            raise

    def truncate_table(self, table_name: str, count_rows: bool = False):
        """
        Truncate all rows from the specified table.

        Parameters:
        - table_name: The SQL Server table to clear (optionally schema-qualified)
        - count_rows: If True, does a COUNT(*) before truncating and returns that.
                      If False, returns None (row count unavailable after TRUNCATE).

        Returns:
        - Number of rows removed if count_rows=True, otherwise None

        Raises:
        - ValueError if table_name is invalid
        - pyodbc.Error (or subclass) on SQL errors
        """

        if not _VALID_TBL.match(table_name):
            logging.error("Invalid table name : %r", table_name)
            raise ValueError(f"Invalid table name :{table_name!r}")

        try:
            with pyodbc.connect(self.conn_str) as conn:
                cursor = conn.cursor()

                rows_before = None
                if count_rows:
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    rows_before = cursor.fetchval()

                cursor.execute(f"TRUNCATE TABLE {table_name}")
                conn.commit()

                logging.info("Truncated table %s", table_name)
                return rows_before

        except Exception as e:
            logging.error("Failed to truncate %s: %s", table_name, e)
            raise

    def _safe_bulk_insert(self, cursor, insert_sql: str, rows: list, df: pd.DataFrame):
        """
        Fallback row-by-row insert to identify exactly which row/values fail.
        """
        for i, params in enumerate(rows):
            try:
                cursor.execute(insert_sql, params)
            except Exception:
                bad = df.iloc[i].to_dict()
                logging.error("Error inserting row %d: %s", i, bad)
                raise

    # Add this method to your SQLServerService class

    def insert_with_generated_id(self, table_name, column_value_dict, id_column="Id"):
        """
        Insert a record with a generated ID by incrementing the maximum existing ID

        Args:
            table_name (str): Name of the target table
            column_value_dict (dict): Dictionary of {column_name: value} pairs to insert (without ID)
            id_column (str): Name of the ID column to generate value for

        Returns:
            int: The generated ID of the inserted record, or None if the operation failed
        """
        if not column_value_dict:
            raise ValueError("No values provided for insertion")

        try:
            with pyodbc.connect(self.conn_str) as conn:
                cursor = conn.cursor()

                # Retrieving the maximum ID
                cursor.execute(f"SELECT ISNULL(MAX({id_column}), 0) FROM {table_name}")
                max_id = cursor.fetchone()[0]
                new_id = max_id + 1

                # Adding the ID to the column values
                insert_data = {id_column: new_id}
                insert_data.update(column_value_dict)

                # Building the columns and params parts of the SQL statement
                columns = list(insert_data.keys())
                placeholders = ["?"] * len(columns)
                values = list(insert_data.values())

                column_str = ", ".join(columns)
                placeholder_str = ", ".join(placeholders)

                # Inserting with explicit values
                insert_sql = f"INSERT INTO {table_name} ({column_str}) VALUES ({placeholder_str})"

                cursor.execute(insert_sql, values)
                conn.commit()

                logging.info(
                    "Inserted record into %s with %s=%d", table_name, id_column, new_id
                )
                return new_id

        except Exception as e:
            logging.error("Failed to insert record into %s: %s", table_name, e)
            return None

    # def bulk_insert_dataframe(
    #     self,
    #     df: pd.DataFrame,
    #     table_name: str,
    #     chunk_size: int = 10_000,
    #     check_trx_id: bool = True,
    # ):
    #     """
    #     Bulk-insert a DataFrame into SQL Server in chunks of up to `chunk_size` rows,
    #     using fast_executemany, with a row-by-row fallback on error.

    #     Parameters:
    #     - df: DataFrame to insert
    #     - table_name: Target SQL Server table
    #     - chunk_size: Number of rows to insert in each batch
    #     - check_trx_id: Whether to check for existing TrxId values before inserting
    #                     Set to False for tables without TrxId or to skip duplicate checking
    #     """
    #     if df.empty:
    #         logging.warning("DataFrame is empty; nothing to insert.")
    #         return

    #     should_check_duplicates = check_trx_id and "TrxId" in df.columns
    #     if check_trx_id and "TrxId" not in df.columns:
    #         logging.warning(
    #             "DataFrame doesn't contain TrxId column; duplicate checking will be skipped."
    #         )

    #     columns = ",".join(df.columns)
    #     placeholders = ",".join("?" for _ in df.columns)
    #     insert_sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
    #     rows = df.values.tolist()

    #     with pyodbc.connect(self.conn_str) as conn:
    #         cursor = conn.cursor()
    #         cursor.fast_executemany = True

    #         for start in range(0, len(rows), chunk_size):
    #             chunk_df = df.iloc[start : start + chunk_size]
    #             batch = rows[start : start + chunk_size]

    #             if should_check_duplicates:

    #                 trx_ids = chunk_df["TrxId"].tolist()

    #                 if trx_ids:

    #                     trx_id_str = ",".join(str(id) for id in trx_ids)
    #                     check_sql = f"SELECT TrxId FROM {table_name} WHERE TrxId IN ({trx_id_str})"

    #                     cursor.execute(check_sql)
    #                     existing_trx_ids = {row.TrxId for row in cursor.fetchall()}

    #                     if existing_trx_ids:

    #                         filtered_indices = chunk_df.index[
    #                             ~chunk_df["TrxId"].isin(existing_trx_ids)
    #                         ]
    #                         filtered_chunk_df = chunk_df.loc[filtered_indices]
    #                         filtered_batch = filtered_chunk_df.values.tolist()

    #                         logging.info(
    #                             "Skipping %d rows with existing TrxIds in rows %d–%d",
    #                             len(existing_trx_ids),
    #                             start,
    #                             start + len(trx_ids) - 1,
    #                         )

    #                         if len(filtered_batch) == 0:
    #                             logging.info(
    #                                 "All rows in this chunk already exist; skipping."
    #                             )
    #                             continue

    #                         batch = filtered_batch

    #             try:
    #                 cursor.executemany(insert_sql, batch)
    #                 conn.commit()
    #                 logging.info(
    #                     "Inserted rows %d–%d into %s",
    #                     start,
    #                     start + len(batch) - 1,
    #                     table_name,
    #                 )
    #             except Exception as batch_err:
    #                 logging.warning(
    #                     "Chunk insert failed at rows %d–%d: %s. Falling back to row-by-row.",
    #                     start,
    #                     start + len(batch) - 1,
    #                     batch_err,
    #                 )
    #                 conn.rollback()

    #                 for i, params in enumerate(batch):

    #                     if should_check_duplicates and "filtered_indices" in locals():
    #                         original_idx = filtered_indices[i]
    #                     else:
    #                         original_idx = chunk_df.index[i]

    #                     try:

    #                         if should_check_duplicates:
    #                             trx_id = df.iloc[original_idx]["TrxId"]
    #                             check_sql = (
    #                                 f"SELECT 1 FROM {table_name} WHERE TrxId = ?"
    #                             )
    #                             cursor.execute(check_sql, [trx_id])
    #                             if cursor.fetchone():
    #                                 logging.info(
    #                                     "Skipping row #%d with existing TrxId: %s",
    #                                     original_idx,
    #                                     trx_id,
    #                                 )
    #                                 continue

    #                         cursor.execute(insert_sql, params)
    #                         conn.commit()
    #                     except Exception as row_err:
    #                         bad = df.iloc[original_idx].to_dict()
    #                         logging.error(
    #                             "Row #%d failed: %s\nData: %s",
    #                             original_idx,
    #                             row_err,
    #                             bad,
    #                         )
    #                         conn.rollback()

    #         logging.info("Bulk insert completed for %s.", table_name)

    def bulk_insert_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        chunk_size: int = 10_000,
        check_trx_id: bool = True,
        upsert_by: list = None,  # New parameter for upsert (e.g., ['ContentId', 'ItemRef'])
    ):
        """
        Bulk-insert or update (upsert) a DataFrame into SQL Server in chunks of up to `chunk_size` rows,
        using fast_executemany, with a row-by-row fallback on error.

        Parameters:
        - df: DataFrame to insert
        - table_name: Target SQL Server table
        - chunk_size: Number of rows to insert in each batch
        - check_trx_id: Whether to check for existing TrxId values before inserting
                        Set to False for tables without TrxId or to skip duplicate checking
        - upsert_by: List of column names to check for existing records (e.g., ['ContentId', 'ItemRef'])
                    If specified, records with matching values will be updated instead of inserted
        """
        if df.empty:
            logging.warning("DataFrame is empty; nothing to insert.")
            return

        should_check_duplicates = check_trx_id and "TrxId" in df.columns
        if check_trx_id and "TrxId" not in df.columns:
            logging.warning(
                "DataFrame doesn't contain TrxId column; duplicate checking will be skipped."
            )

        # Check if we're doing upserts and validate the columns exist
        upsert_mode = upsert_by is not None and len(upsert_by) > 0
        if upsert_mode:
            for col in upsert_by:
                if col not in df.columns:
                    logging.warning(
                        f"Upsert column '{col}' not found in DataFrame; upsert by this column will be skipped."
                    )
                    upsert_by.remove(col)

            if not upsert_by:
                upsert_mode = False
                logging.warning(
                    "No valid upsert columns remain; falling back to insert-only mode."
                )

        columns = ",".join(df.columns)
        placeholders = ",".join("?" for _ in df.columns)
        insert_sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        rows = df.values.tolist()

        with pyodbc.connect(self.conn_str) as conn:
            cursor = conn.cursor()
            cursor.fast_executemany = True

            for start in range(0, len(rows), chunk_size):
                chunk_df = df.iloc[start : start + chunk_size]
                batch = rows[start : start + chunk_size]

                # Lists to store rows for insert and update
                insert_batch = []
                update_rows = []

                # Check for existing records based on TrxId
                if should_check_duplicates:
                    trx_ids = chunk_df["TrxId"].tolist()
                    if trx_ids:
                        trx_id_str = ",".join(str(id) for id in trx_ids)
                        check_sql = f"SELECT TrxId FROM {table_name} WHERE TrxId IN ({trx_id_str})"
                        cursor.execute(check_sql)
                        existing_trx_ids = {row.TrxId for row in cursor.fetchall()}
                        if existing_trx_ids:
                            filtered_indices = chunk_df.index[
                                ~chunk_df["TrxId"].isin(existing_trx_ids)
                            ]
                            filtered_chunk_df = chunk_df.loc[filtered_indices]
                            filtered_batch = filtered_chunk_df.values.tolist()
                            logging.info(
                                "Skipping %d rows with existing TrxIds in rows %d–%d",
                                len(existing_trx_ids),
                                start,
                                start + len(trx_ids) - 1,
                            )
                            if len(filtered_batch) == 0:
                                logging.info(
                                    "All rows in this chunk already exist; skipping."
                                )
                                continue
                            batch = filtered_batch

                # Handle upsert logic if enabled
                if upsert_mode:
                    # Build the WHERE clause to check for existing records
                    upsert_conditions = []
                    for col in upsert_by:
                        # Get unique values for this column in the chunk
                        values = [
                            str(v) if v is not None and not pd.isna(v) else "NULL"
                            for v in chunk_df[col].unique().tolist()
                        ]
                        # Filter out None/NaN values for the IN clause
                        non_null_values = [v for v in values if v != "NULL"]

                        if non_null_values:
                            # Create the condition for this column
                            values_str = ",".join(
                                f"'{v}'" if isinstance(v, str) else str(v)
                                for v in non_null_values
                            )
                            condition = f"{col} IN ({values_str})"
                            upsert_conditions.append(condition)

                    # If we have conditions, check for existing records
                    if upsert_conditions:
                        check_sql = f"SELECT * FROM {table_name} WHERE {' OR '.join(upsert_conditions)}"
                        cursor.execute(check_sql)
                        existing_records = cursor.fetchall()

                        # Create dictionary of existing records keyed by upsert columns
                        existing_dict = {}
                        if existing_records:
                            for record in existing_records:
                                # Create a composite key based on all upsert columns
                                key_values = []
                                for col in upsert_by:
                                    # Get the column index
                                    col_idx = [
                                        i
                                        for i, c in enumerate(cursor.description)
                                        if c[0] == col
                                    ][0]
                                    key_values.append(str(record[col_idx]))

                                key = tuple(key_values)
                                # Store the full record
                                existing_dict[key] = record

                        # Split rows into insert and update batches
                        for i, row in chunk_df.iterrows():
                            # Create the composite key for this row
                            key_values = []
                            for col in upsert_by:
                                val = row[col]
                                key_values.append(
                                    str(val)
                                    if val is not None and not pd.isna(val)
                                    else "NULL"
                                )

                            key = tuple(key_values)

                            # Check if the record exists
                            if key in existing_dict:
                                # This row needs an update
                                update_rows.append((i, row, existing_dict[key]))
                            else:
                                # This row needs an insert
                                insert_batch.append(rows[i - chunk_df.index[0]])
                    else:
                        # No valid conditions, insert everything
                        insert_batch = batch
                else:
                    # Not in upsert mode, insert everything
                    insert_batch = batch

                # Process inserts first
                if insert_batch:
                    try:
                        cursor.executemany(insert_sql, insert_batch)
                        conn.commit()
                        logging.info(
                            "Inserted %d rows into %s",
                            len(insert_batch),
                            table_name,
                        )
                    except Exception as batch_err:
                        logging.warning(
                            "Batch insert failed: %s. Falling back to row-by-row.",
                            batch_err,
                        )
                        conn.rollback()

                        # Row-by-row fallback for inserts
                        for params in insert_batch:
                            try:
                                cursor.execute(insert_sql, params)
                                conn.commit()
                            except Exception as row_err:
                                logging.error(
                                    "Row insert failed: %s",
                                    row_err,
                                )
                                conn.rollback()

                # Process updates
                for idx, row, existing in update_rows:
                    try:
                        # Build the UPDATE statement
                        update_sets = []
                        update_values = []

                        for col in df.columns:
                            # Skip the upsert columns in the SET clause
                            if col not in upsert_by:
                                update_sets.append(f"{col} = ?")
                                update_values.append(row[col])

                        # Add the WHERE conditions
                        where_conditions = []
                        for col in upsert_by:
                            where_conditions.append(f"{col} = ?")
                            update_values.append(row[col])

                        update_sql = f"UPDATE {table_name} SET {', '.join(update_sets)} WHERE {' AND '.join(where_conditions)}"

                        # Execute the update
                        cursor.execute(update_sql, update_values)
                        conn.commit()
                    except Exception as update_err:
                        logging.error(
                            "Row update failed: %s\nData: %s",
                            update_err,
                            row.to_dict(),
                        )
                        conn.rollback()

                if update_rows:
                    logging.info(
                        "Updated %d rows in %s",
                        len(update_rows),
                        table_name,
                    )

            logging.info("Bulk insert/update completed for %s.", table_name)

    def update_column_by_key(
        self,
        table_name: str,
        update_column: str,
        update_value_column: str,
        key_column: str,
        key_value_column: str,
        mapping_df: pd.DataFrame,
        chunk_size: int = 10_000,
    ):
        """
        Update DB Column by key
        """
        if mapping_df.empty:
            logging.warning("Mapping DataFrame is empty; nothing to update.")
            return 0

        total_updates = 0
        update_sql = (
            f"UPDATE {table_name} "
            f"SET    {update_column} = ? "
            f"WHERE  {key_column}   = ?"
        )

        with pyodbc.connect(self.conn_str) as conn:
            cursor = conn.cursor()
            cursor.fast_executemany = True

            # iterating in chunks to bound memory
            for start in range(0, len(mapping_df), chunk_size):
                chunk = mapping_df.iloc[start : start + chunk_size]
                params = list(zip(chunk[update_value_column], chunk[key_value_column]))

                cursor.executemany(update_sql, params)
                conn.commit()
                total_updates += cursor.rowcount

                logging.info(
                    "Chunk %d: updated rows %d–%d",
                    start // chunk_size + 1,
                    start,
                    min(start + chunk_size, len(mapping_df)),
                )

        logging.info("Total rows updated: %d", total_updates)
        return total_updates

    def bulk_update_via_stage(
        self,
        table_name: str,
        update_column: str,
        update_value_column: str,
        key_column: str,
        key_value_column: str,
        mapping_df: pd.DataFrame,
    ):
        # 1) build SQLAlchemy engine (you can reuse across calls)
        engine = sa.create_engine(f"mssql+pyodbc:///?odbc_connect={self.conn_str}")

        # 2) write to temp table
        with engine.begin() as conn:
            mapping_df.to_sql(
                name="#tmp_mapping",
                con=conn,
                index=False,
                if_exists="replace",
                dtype={
                    key_value_column: sa.types.NVARCHAR(length=100),
                    update_value_column: sa.types.NVARCHAR(length=100),
                },
            )

            # 3) single set-based update
            update_sql = f"""
            UPDATE T
            SET T.{update_column} = M.{update_value_column}
            FROM {table_name} AS T
            INNER JOIN #tmp_mapping AS M
                ON T.{key_column} = M.{key_value_column}
            """
            result = conn.execute(sa.text(update_sql))

        return result.rowcount

    def bulk_update_multiple_columns(
        self,
        table_name: str,
        key_columns: list,
        update_df: pd.DataFrame,
        primary_key_columns: list = None,
    ):
        """Update multiple columns in a table using a staging temp table approach"""

        # Building SQLAlchemy engine
        import sqlalchemy as sa

        engine = sa.create_engine(f"mssql+pyodbc:///?odbc_connect={self.conn_str}")

        # If primary key columns not provided, we use default ContentId and FK_ProjectId
        if primary_key_columns is None:
            primary_key_columns = ["ContentId", "FK_ProjectId"]

        # Writing to temp table
        with engine.begin() as conn:
            update_df.to_sql(
                name="#tmp_update_staging", con=conn, index=False, if_exists="replace"
            )

            exclude_columns = set(key_columns + primary_key_columns)
            update_columns = [
                col for col in update_df.columns if col not in exclude_columns
            ]

            if not update_columns:
                logging.warning("No columns to update after excluding keys")
                return 0

            set_clause = ", ".join([f"T.{col} = S.{col}" for col in update_columns])

            join_condition = " AND ".join([f"T.{col} = S.{col}" for col in key_columns])

            update_sql = f"""
            UPDATE T
            SET {set_clause}
            FROM {table_name} AS T
            INNER JOIN #tmp_update_staging AS S
                ON {join_condition}
            """

            result = conn.execute(sa.text(update_sql))

        return result.rowcount

    def export_data_to_ai_db(
        self,
        df: pd.DataFrame,
        stored_procedure: str,
        parameter_name: str,
    ) -> None:
        """
        Sends a pandas DataFrame to a SQL Server stored procedure that
        expects a table-valued parameter (TVP).  No explicit setinputsizes—let
        the driver infer the TVP.
        """
        tvp_rows = [tuple(row) for row in df.itertuples(index=False)]
        call = f"EXEC {stored_procedure} @{parameter_name} = ?"
        with pyodbc.connect(self.conn_str, autocommit=True) as conn:
            cur = conn.cursor()
            cur.execute(call, (tvp_rows,))

    @deprecated(version="1.0.0", reason="Use insert_log instead")
    def insert_log_with_generated_id(
        self, table_name, column_value_dict, id_column="Id"
    ):
        """
        Insert a log record with a generated ID by incrementing the maximum existing ID
        """
        if not _VALID_TBL.match(table_name):
            raise ValueError(f"Invalid table name: {table_name!r}")

        with pyodbc.connect(self.conn_str) as conn:
            cursor = conn.cursor()

            try:

                cursor.execute(f"SELECT ISNULL(MAX({id_column}), 0) FROM {table_name}")
                max_id = cursor.fetchone()[0]
                max_id = int(max_id) if max_id is not None else 0
                new_id = max_id + 1

                insert_data = {id_column: new_id}
                insert_data.update(column_value_dict)

                columns = list(insert_data.keys())
                params = []

                column_str = ", ".join([f"[{col}]" for col in columns])
                placeholder_str = ", ".join(["?" for _ in columns])

                for col in columns:
                    params.append(insert_data[col])

                insert_sql = f"INSERT INTO [{table_name}] ({column_str}) VALUES ({placeholder_str})"

                cursor.execute(insert_sql, params)
                conn.commit()

                return new_id
            except Exception as e:
                conn.rollback()
                print(f"Error in insert_with_generated_id: {e}")
                print(
                    f"SQL: {insert_sql if 'insert_sql' in locals() else 'SQL not built yet'}"
                )
                print(
                    f"Params: {params if 'params' in locals() else 'Params not built yet'}"
                )
                raise

    def insert_log(self, table_name, column_value_dict):
        """
        Insert a log record using SQL Server's IDENTITY column
        """
        if not _VALID_TBL.match(table_name):
            raise ValueError(f"Invalid table name: {table_name!r}")

        if "Id" in column_value_dict:
            del column_value_dict["Id"]

        with pyodbc.connect(self.conn_str) as conn:
            cursor = conn.cursor()
            try:

                columns = list(column_value_dict.keys())
                params = []

                column_str = ", ".join([f"[{col}]" for col in columns])
                placeholder_str = ", ".join(["?" for _ in columns])

                for col in columns:
                    params.append(column_value_dict[col])

                insert_sql = f"INSERT INTO [{table_name}] ({column_str}) VALUES ({placeholder_str})"

                cursor.execute(insert_sql, params)
                conn.commit()

                cursor.execute("SELECT SCOPE_IDENTITY()")
                return cursor.fetchone()[0]

            except Exception as e:
                conn.rollback()
                print(f"Error in insert_log: {e}")
                print(
                    f"SQL: {insert_sql if 'insert_sql' in locals() else 'SQL not built yet'}"
                )
                print(
                    f"Params: {params if 'params' in locals() else 'Params not built yet'}"
                )
                raise

    def export_dataframe_to_stored_procedure(
        self,
        df: pd.DataFrame,
        stored_procedure: str,
        parameter_name: str,
        additional_params: dict = None,
        default_values: dict = None,
    ) -> None:
        """
        Sends a pandas DataFrame to a SQL Server stored procedure that
        expects a table-valued parameter (TVP).

        Parameters:
        - df: DataFrame containing the data to be sent
        - stored_procedure: Name of the stored procedure
        - parameter_name: Name of the TVP parameter in the stored procedure
        - additional_params: Dictionary of additional parameters to pass to the stored procedure
        - default_values: Dictionary of default values to add to each row if columns are missing in df

        Returns:
        - None
        """
        # Make a copy of the DataFrame to avoid modifying the original
        df_copy = df.copy()

        # Add any missing columns with default values
        if default_values:
            for col, value in default_values.items():
                if col not in df_copy.columns:
                    df_copy[col] = value

        # Convert DataFrame to list of tuples
        tvp_rows = [tuple(row) for row in df_copy.itertuples(index=False)]

        # Build the SQL command with parameters
        if additional_params:
            # Create parameter placeholders
            additional_placeholders = ", ".join(
                [f"@{key} = ?" for key in additional_params.keys()]
            )
            call = f"EXEC {stored_procedure} @{parameter_name} = ?, {additional_placeholders}"
            # Create parameters tuple with TVP first, then additional params
            params = (tvp_rows,) + tuple(additional_params.values())
        else:
            call = f"EXEC {stored_procedure} @{parameter_name} = ?"
            params = (tvp_rows,)

        # Execute the stored procedure
        with pyodbc.connect(self.conn_str, autocommit=True) as conn:
            cur = conn.cursor()
            cur.execute(call, params)

        logging.info(f"Executed {stored_procedure} with {len(tvp_rows)} rows")

    def export_deals_data_to_raw(
        self, deals_data: pd.DataFrame, fk_project_id: int
    ) -> None:
        """
        Export deals data to Deals_RAW table using the sp_Copy_AIDetailedDeals stored procedure.

        Parameters:
        - deals_data (pd.DataFrame): DataFrame containing deals data with appropriate columns
        - fk_project_id (int): Project ID to use for all records

        Returns:
        - None
        """
        if deals_data.empty:
            logging.warning("Deals DataFrame is empty; nothing to export.")
            return

        # Map DataFrame columns to match the expected columns for Type_Deals_Detailed
        column_mapping = {
            "ContentId": "ContentId",
            # FK_ProjectId will be set explicitly
            "EntityId": "EntityId",
            "EntityName": "EntityName",
            "Title": "Title",
            "FK_StatusId": "FK_StatusId",
            "Available": "Available",
            "DealType": "DealType",
            "Description": "Description",
            "Location": "Location",
            "Points": "Points",
            "Website": "Website",
            "Categories": "ContentGroup",  # Assuming ContentGroup corresponds to Categories
            "Latitude": "Latitude",
            "Longitude": "Longitude",
            "MCC": "MCC",
            "Published": "Published",
            "ItemRef": "ItemRef",
            "EntityDescription": "EntityDescription",
        }

        # Prepare the DataFrame for the stored procedure
        df_to_export = pd.DataFrame()

        for target_col, source_col in column_mapping.items():
            if source_col in deals_data.columns:
                df_to_export[target_col] = deals_data[source_col]
            else:
                df_to_export[target_col] = None

        # Set FK_ProjectId explicitly for all rows
        df_to_export["FK_ProjectId"] = fk_project_id

        # Ensure all required columns are present with correct types
        required_columns = [
            "ContentId",
            "FK_ProjectId",
            "EntityId",
            "EntityName",
            "Title",
            "FK_StatusId",
            "Available",
            "DealType",
            "Description",
            "Location",
            "Points",
            "Website",
            "Categories",
            "Latitude",
            "Longitude",
            "MCC",
            "Published",
            "ItemRef",
            "EntityDescription",
        ]

        for col in required_columns:
            if col not in df_to_export.columns:
                df_to_export[col] = None

        # Enforce specific data types to match SQL expectations
        # Integer columns (excluding MCC which should be a string)
        int_columns = [
            "ContentId",
            "FK_ProjectId",
            "EntityId",
            "FK_StatusId",
            "Points",
            "Published",
        ]
        for col in int_columns:
            df_to_export[col] = (
                pd.to_numeric(df_to_export[col], errors="coerce").fillna(0).astype(int)
            )

        # String columns - ensure they're all strings and handle NaN/None
        str_columns = [col for col in df_to_export.columns if col not in int_columns]
        for col in str_columns:
            df_to_export[col] = df_to_export[col].fillna("").astype(str)
            # Special handling for Location which cannot be NULL
            if col == "Location" and df_to_export[col].isnull().any():
                df_to_export[col] = df_to_export[col].fillna("")

        # Limit string lengths according to table definition
        length_limits = {
            "EntityName": 250,
            "Available": 250,
            "DealType": 250,
            "Location": 350,
            "Categories": 1000,
            "Latitude": 250,
            "Longitude": 250,
            "MCC": 50,  # MCC is a string field
            "ItemRef": 50,
        }

        for col, limit in length_limits.items():
            if col in df_to_export.columns:
                df_to_export[col] = df_to_export[col].str.slice(0, limit)

        try:
            # Direct approach using pyodbc to better control the TVP binding
            tvp_rows = [
                tuple(row)
                for row in df_to_export[required_columns].itertuples(index=False)
            ]

            with pyodbc.connect(self.conn_str) as conn:
                cursor = conn.cursor()
                # Execute the stored procedure with the table-valued parameter
                cursor.execute("{CALL sp_Copy_AIDetailedDeals(?)}", (tvp_rows,))
                conn.commit()

            logging.info(
                f"Successfully exported {len(df_to_export)} deals to Deals_RAW table"
            )
        except Exception as e:
            logging.error(f"Failed to export deals data: {e}")
            logging.error(f"DataFrame column types: {df_to_export.dtypes}")
            # Print a sample row to help with debugging
            if not df_to_export.empty:
                logging.error(f"Sample row: {df_to_export.iloc[0].to_dict()}")
            raise
