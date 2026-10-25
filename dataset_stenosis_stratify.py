import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from scipy.stats import ttest_ind

if __name__ == "__main__":
    input_csv = "/home/b09401064/pelvic_CTA/nnUNet_code/csv/dataset_stenosis_stratify_input.csv"
    train_csv = "/home/b09401064/pelvic_CTA/nnUNet_code/csv/train.csv"
    test_csv = "/home/b09401064/pelvic_CTA/nnUNet_code/csv/test.csv"
    t_test_*csv = "/home/b09401064/pelvic_CTA/nnUNet_code/csv/t_test_results.csv"
    N_SPLIT = 5
    N_CANDIDATES = 100

    df = pd.read_csv(input_csv)

    id_column = "RENAMED_ID"
    has_report_column = "HAS_CATH_REPORT"
    has_pta_column = "PTA"
    stenosis_columns = [column for column in df.columns if column not in [id_column, has_report_column, has_pta_column]]
    print("stenosis_columns", stenosis_columns, flush=True)

    # Convert invalid values such as "N" to missing values.
    df[stenosis_columns] = df[stenosis_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    # Remove blank rows, but retain valid patients even when all stenosis
    # measurements are missing; no-report patients still belong in training.
    df = df[df[id_column].notna()]
    df = df.reset_index(drop=True)

    # if the patient has no report, then the data would automatically be assigned to the train set
    # because we can't compare stenosis detection results to the report. 
    # So we will remove these patients from the dataset before stratification.
    df_to_remove = df[df[has_report_column] == 0].reset_index(drop=True)
    df_to_stratify = df[df[has_report_column] == 1].reset_index(drop=True)

    # Assumption: an empty stenosis cell means 50% stenosis.
    all_values = df[stenosis_columns].fillna(50)
    stratify_values = df_to_stratify[stenosis_columns].fillna(50)

    def make_labels(values):
        return np.concatenate(
            [
                (values >= 50).astype(int).to_numpy(),
                (values >= 70).astype(int).to_numpy(),
            ],
            axis=1,
        )

    all_labels = make_labels(all_values)
    labels = make_labels(stratify_values)
    fixed_train_mask = df[has_report_column].eq(0).to_numpy()
    fixed_train_features = np.concatenate(
        [all_values.to_numpy()[fixed_train_mask], all_labels[fixed_train_mask]],
        axis=1,
    )
    stratify_features = np.concatenate(
        [stratify_values.to_numpy(), labels],
        axis=1,
    )
    indices = np.arange(len(df_to_stratify)).reshape(-1, 1)

    # No-report patients are all assigned to training. Therefore calculate
    # the test count from the full dataset, not just the report subset.
    desired_test_count = round(len(df) / N_SPLIT)
    if len(df_to_stratify) < 2:
        raise ValueError("At least two patients with reports are required to split the dataset.")
    if not 1 <= desired_test_count < len(df_to_stratify):
        raise ValueError(
            f"The requested {N_SPLIT - 1}:{1} overall split cannot be made because too many patients "
            "lack reports."
        )

    # Try multiple valid splits and choose the one whose final train/test
    # feature distributions are closest. The fixed no-report rows are part
    # of the train distribution during scoring.
    best_score = np.inf
    best_split = None
    for random_state in range(N_CANDIDATES):
        splitter = MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=desired_test_count,
            random_state=random_state,
        )
        candidate_train, candidate_test = next(splitter.split(indices, labels))
        candidate_train_features = np.concatenate(
            [fixed_train_features, stratify_features[candidate_train]],
            axis=0,
        )
        candidate_test_features = stratify_features[candidate_test]
        score = np.abs(
            candidate_train_features.mean(axis=0)
            - candidate_test_features.mean(axis=0)
        ).mean()
        if score < best_score:
            best_score = score
            best_split = candidate_train, candidate_test

    train_indices, test_indices = best_split

    train_df = pd.concat(
        [df_to_stratify.iloc[train_indices], df_to_remove],
        ignore_index=True,
    )
    train_df.sort_values(by=id_column, inplace=True)
    test_df = df_to_stratify.iloc[test_indices].copy()

    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print(f"Train patients: {len(train_df)}")
    print(f"Test patients: {len(test_df)}")
    print(f"Mean train/test distribution difference: {best_score:.4f}")
    
    print("--RENAMED_ID--")
    print("Train:", train_df[id_column].to_numpy().astype(int).tolist())
    print("Test:", test_df[id_column].to_numpy().astype(int).tolist())

    # Perform t tests to verify that the train/test distributions are not significantly different.
    # and save the results to a csv file.
    t_test_results = []
    for column in stenosis_columns:
        train_values = train_df[column].dropna()
        test_values = test_df[column].dropna()
        if len(train_values) < 2 or len(test_values) < 2:
            t_statistic, p_value = np.nan, np.nan
        else:
            t_statistic, p_value = ttest_ind(
                train_values, test_values, equal_var=False
            )
        t_test_results.append(
            {
                "column": column,
                "train_mean": train_values.mean(),
                "test_mean": test_values.mean(),
                "t_statistic": t_statistic,
                "p_value": p_value,
            }
        )
    for column in [has_report_column, has_pta_column]:
        train_values = train_df[column].dropna()
        test_values = test_df[column].dropna()
        if len(train_values) < 2 or len(test_values) < 2:
            t_statistic, p_value = np.nan, np.nan
        else:
            t_statistic, p_value = ttest_ind(
                train_values, test_values, equal_var=False
            )
        t_test_results.append(
            {
                "column": column,
                "train_mean": train_values.mean(),
                "test_mean": test_values.mean(),
                "t_statistic": t_statistic,
                "p_value": p_value,
            }
        )
    t_test_df = pd.DataFrame(t_test_results)
    t_test_df.to_csv(t_test_csv, index=False)
    print("T-test results saved to", t_test_csv, flush=True)