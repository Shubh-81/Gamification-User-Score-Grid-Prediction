import pandas as pd
from xgboost import XGBRegressor
import copy
import argparse
import datetime


def parse_arguments():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('csv_file', type=str, help='Path to the input CSV file')
    parser.add_argument('div', type=int, help='Number of divisions for grid')
    parser.add_argument('pad', type=int, help='Percentage to pad the user score')
    return parser.parse_args()


def load_data(file_path):
    # Load CSV data into a DataFrame
    return pd.read_csv(file_path)


def clean_data(df):
    # Clean and preprocess DataFrame, extracting date information
    final_df = df[['id_coroebus_group',
                   'id_coroebus_team', 'id_coroebus_user', 'score', 'id_role', 'unit_name', 'updated_date_time']]
    final_df['updated_date_time'] = pd.to_datetime(final_df['updated_date_time'])
    final_df = final_df.sort_values(by='updated_date_time')

    last_date = final_df['updated_date_time'].max()
    next_30_days = pd.date_range(start=last_date + datetime.timedelta(days=1), periods=30, freq='D')
    date_df = pd.DataFrame(next_30_days, columns=['updated_date_time'])

    final_df['year'] = final_df['updated_date_time'].apply(lambda x: x.year)
    final_df['month'] = final_df['updated_date_time'].apply(lambda x: x.month)
    final_df['day'] = final_df['updated_date_time'].apply(lambda x: x.day)
    final_df['hour'] = final_df['updated_date_time'].apply(lambda x: x.hour)
    final_df['minute'] = final_df['updated_date_time'].apply(lambda x: x.minute)
    final_df.drop('updated_date_time', axis=1, inplace=True)

    return final_df, date_df


def create_ranges(row, num_divisions=5, pad_percentage=5):
    # Create score ranges for each row
    min_score = row['min_score']
    max_score = row['max_score']
    unit_name = row['unit_name']

    min_score = int(min_score + min_score * pad_percentage / 100)
    max_score = int(max_score + 2 * max_score * pad_percentage / 100)

    range_width = (max_score - min_score) / num_divisions

    new_rows = []
    for i in range(num_divisions):
        start_range = int(min_score + i * range_width)
        end_range = int(min_score + (i + 1) * range_width)

        new_row = {
            'id_coroebus_group': row['id_coroebus_group'],
            'id_coroebus_team': row['id_coroebus_team'],
            'id_coroebus_user': row['id_coroebus_user'],
            'id_role': row['id_role'],
            'unit_name': unit_name,
            'grid_index': i + 1,
            'min_score': start_range,
            'max_score': end_range,
        }

        new_rows.append(new_row)

    return pd.DataFrame(new_rows)


def create_output_dataframe(input_df, date_df):
    # Create dataframe on which model will give prediction
    output_df = copy.deepcopy(input_df)
    output_df['year'] = 0
    output_df['month'] = 0
    output_df['day'] = 0
    output_df['hour'] = 0
    output_df['minute'] = 0
    output_df.drop_duplicates(inplace=True)
    output_df.drop(['year', 'month', 'day', 'hour', 'minute'], axis=1, inplace=True)
    output_df = output_df.merge(date_df, how='cross')
    output_df.drop_duplicates(inplace=True)
    output_df, _ = clean_data(output_df)
    output_df = pd.get_dummies(output_df, columns=['unit_name'])
    output_df.drop('score', inplace=True, axis=1)
    return output_df


def postprocess_output(output_df, args):
    # Postprocess the output from model in the required format
    output_df['score'] = output_df['score'].apply(lambda x: x + x * args.pad / 100)
    output_df['score'] = output_df['score'].apply(lambda x: round(x))
    output_df['score'] = output_df['score'].apply(lambda x: 1 if x <= 0 else x)
    output_df['unit_name'] = output_df.filter(like='unit_name').idxmax(axis=1)
    output_df['unit_name'] = output_df['unit_name'].apply(lambda x: x.split('_')[2])
    output_df.drop(columns=output_df.filter(like='unit_name_').columns, inplace=True)

    preds_min = output_df.groupby(['id_coroebus_user', 'unit_name'])['score'].min().reset_index()
    preds_max = output_df.groupby(['id_coroebus_user', 'unit_name'])['score'].max().reset_index()
    output_df = output_df.merge(preds_min, how='left', on=['id_coroebus_user', 'unit_name'])
    output_df = output_df.merge(preds_max, how='left', on=['id_coroebus_user', 'unit_name'])
    output_df.rename(columns={'score_x': 'score', 'score_y': 'min_score', 'score': 'max_score'}, inplace=True)
    output_df.drop(['year', 'month', 'day', 'hour', 'minute', 'score'], axis=1, inplace=True)
    output_df.drop_duplicates(inplace=True)

    result_df = pd.concat([create_ranges(row, args.div, args.pad) for _, row in output_df.iterrows()],
                          ignore_index=True)
    return result_df


def main():
    # Entry point of the script
    args = parse_arguments()
    input_df = load_data(args.csv_file)

    input_df, date_df = clean_data(input_df)
    output_df = create_output_dataframe(input_df, date_df)
    input_df = pd.get_dummies(input_df, columns=['unit_name'])

    model = XGBRegressor()
    X = input_df.drop('score', axis=1)
    y = input_df['score']
    model.fit(X, y)

    output_df['score'] = model.predict(output_df)
    result_df = postprocess_output(output_df, args)
    result_df.to_csv('predictions.csv', index=False)
    print('Predictions saved to predictions.csv')


if __name__ == "__main__":
    main()
