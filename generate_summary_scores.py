import os
import glob
import argparse
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

from melanoma.evaluation.summary import generate_df_scores


def main(args):
    model_dir = args.model_dir
    keyword = args.keyword
    output_dir = args.output_dir
    exp_dirs = glob.glob(os.path.join(model_dir, f'*{keyword}*'))
    print('Generating summary table.')
    df_scores = generate_df_scores(exp_dirs)
    df_scores = df_scores.sort_values(args.sort_by, ascending=args.ascending).reset_index(drop=True)
    print(df_scores)

    filepath = os.path.join(output_dir, 'scores.csv')
    df_scores.to_csv(filepath, index=False)
    print(f'Saved summary table to {filepath}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir',
                        '-m',
                        default='models',
                        type=str,
                        help='Path to model directory.')
    parser.add_argument('--keyword',
                        '-k',
                        default='exp',
                        type=str,
                        help='Keyword to filter experiment directories')
    parser.add_argument('--output_dir',
                        '-o',
                        default='output',
                        type=str,
                        help='Path to model directory.')
    parser.add_argument('--sort_by',
                        '-s',
                        default='val_auc_score',
                        type=str,
                        help='Colname to sort table by.')
    parser.add_argument('--ascending',
                        '-a',
                        default=False,
                        action='store_true',
                        help='Sort table in ascending order by `sort_by`.')
    #parser.add_argument('--experiment',
    #                    '-e',
    #                    default=None,
    #                    action=str,
    #                    help='Experiment name to subset')
    args = parser.parse_args()

    main(args)
