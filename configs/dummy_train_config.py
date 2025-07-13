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
    text_embedding_params:
        enabled: true
        text_column: 'text_notes'
        model_name: 'emilyalsentzer/Bio_ClinicalBERT'
        pooling_strategy: 'cls'
        max_length: 512
        batch_size: 32
    ontology_embedding_params:
        enabled: true
        code_columns: ['icdCode']
        ontology_data: 'data/ontology.json'
        node2vec_params:
            dimensions: 64
            walk_length: 30
            num_walks: 200
            workers: 4
        save_path: 'outputs/ontology_embeddings.emb'
    optuna:
        lgbm:
            use_optuna: true
            n_trials: 10
        xgboost_meta:
            use_optuna_for_meta: true
            n_trials: 10
    balancing:
        use_rsmote_gan_in_cv: true
    """)
