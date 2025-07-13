import yaml

def get_dummy_config():
    return yaml.safe_load("""
    use_dummy_data_for_full_run: false
    los_column: 'lengthofStay'
    data_dir: 'data/'
    data_paths:
        train: 'trainData.csv'
        val: 'valData.csv'
        test: 'testData.csv'
        target_column: 'outcomeType'
    preprocessing:
        numerical_cols:
            - 'patientAge'
            - 'hematocrit'
            - 'hemoglobin'
            - 'leucocitos'
            - 'lymphocytes'
            - 'urea'
            - 'creatinine'
            - 'platelets'
            - 'diuresis'
        categorical_cols:
            - 'requestType'
            - 'requestBedType'
            - 'admissionBedType'
            - 'admissionHealthUnit'
            - 'patientGender'
            - 'patientFfederalUnit'
            - 'icdCode'
            - 'blodPressure'
            - 'glasgowScale'
    ensemble:
        n_outer_folds: 2
        n_inner_folds_for_oof: 2
        train_lgbm: true
        lgbm_params:
            num_boost_round: 10
            early_stopping_rounds: 5
        train_teco: false
        train_gnn: false
        train_meta_learner: true
        meta_learner_xgb_params:
            num_boost_round: 10
            early_stopping_rounds: 5
    optuna:
        lgbm:
            use_optuna: false
        xgboost_meta:
            use_optuna_for_meta: false
    balancing:
        use_rsmote_gan_in_cv: false
    """)
