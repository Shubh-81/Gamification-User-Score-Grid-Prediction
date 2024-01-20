# User Score Grid Prediction for the Next 30 Days

## Overview

This Python script utilizes the XGBoost algorithm to predict user score grids for the next 30 days based on historical data obtained from the `db_coroebus_tgc` database table `tbl_coroebus_kpi_data`. The script performs data cleaning, feature engineering, and model training to generate predictions.

## Requirements

- Python 3.x
- pandas
- xgboost
- argparse
- datetime

To install the required dependencies, you can use the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the script with the following command:

   ```bash
   python script_name.py csv_file div pad
   ```

   - `csv_file`: Path to the input CSV file containing historical data.
   - `div`: Number of divisions for the score grid.
   - `pad`: Percentage to pad the user score.

   Example:

   ```bash
   python script.py data.csv 5 10
   ```

2. The script will generate predictions and save the results in a file named `predictions.csv`.

## Input Data

The input data is expected to be in the format of the `tbl_coroebus_kpi_data` table with columns:

- `id_coroebus_group`
- `id_coroebus_team`
- `id_coroebus_user`
- `score`
- `id_role`
- `unit_name`
- `updated_date_time`

## Output

The script generates predictions for the next 30 days, creating a user score grid based on the specified divisions and padding percentage. The results are saved in the `predictions.csv` file.

## Predictions Output Columns

The `predictions.csv` file has the following columns:

1. **id_coroebus_group**: Unique identifier for a group in the `tbl_coroebus_kpi_data` table.
2. **id_coroebus_team**: Unique identifier for a team in the `tbl_coroebus_kpi_data` table.
3. **id_coroebus_user**: Unique identifier for a user in the `tbl_coroebus_kpi_data` table.
4. **id_role**: Identifier for the user's role.
5. **unit_name**: Name of the unit associated with the user.
6. **grid_index**: Index of the score grid. The script divides the score range into multiple grids, and this index represents the specific grid.
7. **min_score**: The minimum score for the given grid index. This is calculated based on the historical data and padding.
8. **max_score**: The maximum score for the given grid index. This is also calculated based on the historical data and padding.

These columns provide information about the predicted user score grids for a specific user, role, and unit over the next 30 days. The grid_index allows you to identify the range of scores within each grid, and the min_score and max_score columns define the boundaries of the score range for that grid.
