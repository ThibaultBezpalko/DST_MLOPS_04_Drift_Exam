import evidently
import datetime
import pandas as pd
import numpy as np
import requests
import zipfile
import io
import json

from sklearn import datasets, ensemble, model_selection
from scipy.stats import anderson_ksamp

from evidently.metrics import RegressionQualityMetric, RegressionErrorPlot, RegressionErrorDistribution
from evidently.metric_preset import DataDriftPreset, RegressionPreset, TargetDriftPreset
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report
from evidently.ui.workspace import Workspace

# ignore warnings
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

############################
## INGESTION 
############################

# custom functions
def _fetch_data() -> pd.DataFrame:
    content = requests.get("https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip", verify=False).content
    with zipfile.ZipFile(io.BytesIO(content)) as arc:
        raw_data = pd.read_csv(arc.open("hour.csv"), header=0, sep=',', parse_dates=['dteday']) 
    return raw_data

def _process_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    raw_data.index = raw_data.apply(lambda row: datetime.datetime.combine(row.dteday.date(), datetime.time(row.hr)),
                                    axis=1)
    return raw_data


############################
## HTML REPORT GENERATION
############################

# # Initialize the report with desired metrics
# data_drift_dataset_report = Report(metrics=[
#     DatasetDriftMetric(),
#     DataDriftTable(),    
# ])

# # Run the report
# data_drift_dataset_report.run(reference_data=adult_ref, 
#                                 current_data=adult_cur)

# # Convert the JSON string to a Python dictionary for pretty printing
# report_data = json.loads(data_drift_dataset_report.json())

# # Save the report in JSON format with indentation for better readability
# with open('data_drift_report.json', 'w') as f:
#     json.dump(report_data, f, indent=4)

# # save HTML
# data_drift_dataset_report.save_html("Classification Report.html")

# # in a notebook run :
# # data_drift_dataset_report.show()


############################
## EVIDENTLY FUNCTIONS
############################

def add_report_to_workspace(workspace, project_name, project_description, report):
    """
    Adds a report to an existing or new project in a workspace.
    This function will be useful to you
    """
    # Check if project already exists
    project = None
    for p in workspace.list_projects():
        if p.name == project_name:
            project = p
            break

    # Create a new project if it doesn't exist
    if project is None:
        project = workspace.create_project(project_name)
        project.description = project_description

    # Add report to the project
    workspace.add_report(project.id, report)
    print(f"New report added to project {project_name}")


############################
## MAIN
############################

# Common data drift metric
# DatasetDriftMetric(),
# A line of analysis for each column
# DataDriftTable(),    


if __name__ == "__main__":
    WORKSPACE_NAME = "datascientest-workspace"
    PROJECT_NAME = "drift_monitoring_exam"
    PROJECT_DESCRIPTION = "Evidently Dashboards"

    # load data
    print("=== Loading data ===")
    raw_data = _process_data(_fetch_data())

    ############################
    ## MODEL TRAIN
    ############################

    # Feature selection
    target = 'cnt'
    prediction = 'prediction'
    numerical_features = ['temp', 'atemp', 'hum', 'windspeed', 'mnth', 'hr', 'weekday']
    categorical_features = ['season', 'holiday', 'workingday']

    # Reference and current data split
    reference_jan11 = raw_data.loc['2011-01-01 00:00:00':'2011-01-28 23:00:00']
    current_feb11 = raw_data.loc['2011-01-29 00:00:00':'2011-02-28 23:00:00']

    # print(f'reference_jan11 : {reference_jan11.shape}')
    # print(f'current_feb11 : {current_feb11.shape}')

    # Train test split ONLY on reference_jan11
    print("=== Train, test, split on January data ===")
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        reference_jan11[numerical_features + categorical_features],
        reference_jan11[target],
        test_size=0.3
    )

    # Model training
    print("=== Fitting the random forest regressor on train data ===")
    regressor = ensemble.RandomForestRegressor(random_state = 0, n_estimators = 50)
    regressor.fit(X_train, y_train)

    # Predictions
    preds_train = regressor.predict(X_train)
    preds_test = regressor.predict(X_test)


    ############################
    ## MODEL VALIDATION
    ############################

    # Add actual target and prediction columns to the training data for later performance analysis
    X_train['target'] = y_train
    X_train['prediction'] = preds_train

    # Add actual target and prediction columns to the test data for later performance analysis
    X_test['target'] = y_test
    X_test['prediction'] = preds_test

    # Initialize the column mapping object which evidently uses to know how the data is structured
    column_mapping_01 = ColumnMapping()

    # Map the actual target and prediction column names in the dataset for evidently
    column_mapping_01.target = 'target'
    column_mapping_01.prediction = 'prediction'

    # Specify which features are numerical and which are categorical for the evidently report
    column_mapping_01.numerical_features = numerical_features
    column_mapping_01.categorical_features = categorical_features

    # Generate a tag for the bunch of reports
    tag = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S") # Example: '2024-12-25T14:30:15'
    print(f'=== Generation tag: {tag} ===')  

    # Initialize the regression performance report with the default regression metrics preset
    validation_report_01 = Report(
        metrics=[
            RegressionPreset(),
        ],
        metadata = {
            "name": "drift_exam_01_model_validation",
        },
        tags=[tag],
    )

    # Run the regression performance report using the training data as reference and test data as current
    # The data is sorted by index to ensure consistent ordering for the comparison
    print("=== Running model validation report on January data ===")
    validation_report_01.run(reference_data=X_train.sort_index(), 
                                    current_data=X_test.sort_index(),
                                    column_mapping=column_mapping_01)


    # Convert the JSON string to a Python dictionary for pretty printing
    report_data_01 = json.loads(validation_report_01.json())

    # Save the report in JSON format with indentation for better readability
    with open('drift_exam_01_model_validation.json', 'w') as f:
        json.dump(report_data_01, f, indent=4)

    # save HTML
    validation_report_01.save_html("drift_exam_01_model_validation.html")

    # # Generate the regression performance report
    # regression_report = generate_regression_report(reference_data, current_data)

    # Create and Add report to workspace
    workspace = Workspace.create(WORKSPACE_NAME)
    add_report_to_workspace(workspace, PROJECT_NAME, PROJECT_DESCRIPTION, validation_report_01)

    
    ##########################################
    ## ANALYSE DERIVE MODELE - Janvier
    ##########################################

    # Train the production model
    print("=== Training model on complete January data ===")
    regressor.fit(reference_jan11[numerical_features + categorical_features], reference_jan11[target])

    # Perform column mapping
    column_mapping = ColumnMapping()
    column_mapping.target = target
    column_mapping.prediction = prediction
    column_mapping.numerical_features = numerical_features
    column_mapping.categorical_features = categorical_features

    # Generate predictions for the reference data
    ref_prediction_jan11 = regressor.predict(reference_jan11[numerical_features + categorical_features])
    reference_jan11['prediction'] = ref_prediction_jan11

    # Initialize the regression performance report with the default regression metrics preset
    production_report_02 = Report(
        metrics=[
            RegressionPreset(),
        ],
        metadata = {
            "name": "drift_exam_02_model_drift_jan",
        },
        tags=[tag],
    )

    # Run the regression performance report using the reference data
    print("=== Running model drift report on January data ===")
    production_report_02.run(
        reference_data=None, 
        current_data=reference_jan11,
        column_mapping=column_mapping
    )

    # Convert the JSON string to a Python dictionary for pretty printing
    report_data_02 = json.loads(production_report_02.json())

    # Save the report in JSON format with indentation for better readability
    with open('drift_exam_02_model_drift_jan.json', 'w') as f:
        json.dump(report_data_02, f, indent=4)

    # save HTML
    production_report_02.save_html("drift_exam_02_model_drift_jan.html")

    # # Generate the regression performance report
    # regression_report = generate_regression_report(reference_data, current_data)

    # Add report to workspace
    add_report_to_workspace(workspace, PROJECT_NAME, PROJECT_DESCRIPTION, production_report_02)



    ##############################################
    ## ANALYSE DERIVE MODELE - Février Week 1
    ##############################################

    # Generate predictions for the current data
    print("=== Predicting on February 1st week data ===")
    current_feb11_01 = current_feb11.loc['2011-01-29 00:00:00':'2011-02-07 23:00:00']
    cur_prediction_feb11_01 = regressor.predict(current_feb11_01[numerical_features + categorical_features])
    current_feb11_01['prediction'] = cur_prediction_feb11_01

    # Initialize the regression performance report with the default regression metrics preset
    production_report_03 = Report(
        metrics=[
            RegressionPreset(),
        ],
        metadata = {
            "name": "drift_exam_03_model_drift_feb_01",
        },
        tags=[tag],
    )

    # Run the regression performance report using the reference data
    print("=== Running model drift report on February 1st week data ===")
    production_report_03.run(
        reference_data=reference_jan11, 
        current_data=current_feb11_01,
        column_mapping=column_mapping
    )

    # Convert the JSON string to a Python dictionary for pretty printing
    report_data_03 = json.loads(production_report_03.json())

    # Save the report in JSON format with indentation for better readability
    with open('drift_exam_03_model_drift_feb_01.json', 'w') as f:
        json.dump(report_data_03, f, indent=4)

    # save HTML
    production_report_03.save_html("drift_exam_03_model_drift_feb_01.html")

    # # Generate the regression performance report
    # regression_report = generate_regression_report(reference_data, current_data)

    # Add report to workspace
    add_report_to_workspace(workspace, PROJECT_NAME, PROJECT_DESCRIPTION, production_report_03)


    ##############################################
    ## ANALYSE DERIVE MODELE - Février Week 2
    ##############################################

    # Generate predictions for the current data
    print("=== Predicting on February 2nd week data ===")
    current_feb11_02= current_feb11.loc['2011-02-08 00:00:00':'2011-02-14 23:00:00']
    cur_prediction_feb11_02 = regressor.predict(current_feb11_02[numerical_features + categorical_features])
    current_feb11_02['prediction'] = cur_prediction_feb11_02

    # Initialize the regression performance report with the default regression metrics preset
    production_report_04 = Report(
        metrics=[
            RegressionPreset(),
        ],
        metadata = {
            "name": "drift_exam_04_model_drift_feb_02",
        },
        tags=[tag],
    )

    # Run the regression performance report using the reference data
    print("=== Running model drift report on February 2nd week data ===")
    production_report_04.run(
        reference_data=reference_jan11, 
        current_data=current_feb11_02,
        column_mapping=column_mapping
    )

    # Convert the JSON string to a Python dictionary for pretty printing
    report_data_04 = json.loads(production_report_04.json())

    # Save the report in JSON format with indentation for better readability
    with open('drift_exam_04_model_drift_feb_02.json', 'w') as f:
        json.dump(report_data_04, f, indent=4)

    # save HTML
    production_report_04.save_html("drift_exam_04_model_drift_feb_02.html")

    # # Generate the regression performance report
    # regression_report = generate_regression_report(reference_data, current_data)

    # Add report to workspace
    add_report_to_workspace(workspace, PROJECT_NAME, PROJECT_DESCRIPTION, production_report_04)


    ##############################################
    ## ANALYSE DERIVE MODELE - Février Week 3
    ##############################################

    # Generate predictions for the current data
    print("=== Predicting on February 3rd week data ===")
    current_feb11_03 = current_feb11.loc['2011-02-15 00:00:00':'2011-02-21 23:00:00']
    cur_prediction_feb11_03 = regressor.predict(current_feb11_03[numerical_features + categorical_features])
    current_feb11_03['prediction'] = cur_prediction_feb11_03

    # Initialize the regression performance report with the default regression metrics preset
    production_report_05 = Report(
        metrics=[
            RegressionPreset(),
        ],
        metadata = {
            "name": "drift_exam_05_model_drift_feb_03",
        },
        tags=[tag],
    )

    # Run the regression performance report using the reference data
    print("=== Running model drift report on February 3rd week data ===")
    production_report_05.run(
        reference_data=reference_jan11, 
        current_data=current_feb11_03,
        column_mapping=column_mapping
    )

    # Convert the JSON string to a Python dictionary for pretty printing
    report_data_05 = json.loads(production_report_05.json())

    # Save the report in JSON format with indentation for better readability
    with open('drift_exam_05_model_drift_feb_03.json', 'w') as f:
        json.dump(report_data_05, f, indent=4)

    # save HTML
    production_report_05.save_html("drift_exam_05_model_drift_feb_03.html")

    # # Generate the regression performance report
    # regression_report = generate_regression_report(reference_data, current_data)

    # Add report to workspace
    add_report_to_workspace(workspace, PROJECT_NAME, PROJECT_DESCRIPTION, production_report_05)


 
    ############################################################
    ## ANALYSE DERIVE MODELE - Février Worst Week (3rd)
    ############################################################

    # Initialize the regression performance report with the default regression metrics preset
    production_report_06 = Report(
        metrics=[
            TargetDriftPreset(),
        ],
        metadata = {
            "name": "drift_exam_06_target_drift_feb_03",
        },
        tags=[tag],
    )

    # Run the regression performance report using the reference data
    print("=== Running model drift report on February 3rd week data ===")
    production_report_06.run(
        reference_data=reference_jan11, 
        current_data=current_feb11_03,
        column_mapping=column_mapping
    )

    # Convert the JSON string to a Python dictionary for pretty printing
    report_data_06 = json.loads(production_report_06.json())

    # Save the report in JSON format with indentation for better readability
    with open('drift_exam_06_target_drift_feb_03.json', 'w') as f:
        json.dump(report_data_06, f, indent=4)

    # save HTML
    production_report_06.save_html("drift_exam_06_target_drift_feb_03.html")

    # # Generate the regression performance report
    # regression_report = generate_regression_report(reference_data, current_data)

    # Add report to workspace
    add_report_to_workspace(workspace, PROJECT_NAME, PROJECT_DESCRIPTION, production_report_06)

    
    ######################################################
    ## ANALYSE DERIVE DONNEES - Février Week 3
    ######################################################

    # Perform column mapping
    column_mapping_drift = ColumnMapping()

    column_mapping_drift.target = target
    column_mapping_drift.prediction = prediction
    column_mapping_drift.numerical_features = numerical_features
    column_mapping_drift.categorical_features = []

    # Initialize the data drift report
    data_drift_report_07 = Report(
        metrics=[
            DataDriftPreset(),
        ],
        metadata = {
            "name": "drift_exam_07_data_drift_feb_03",
        },
        tags=[tag],
    )

    print("=== Running data drift report on February 3rd week data ===")
    data_drift_report_07.run(
        reference_data=reference_jan11,
        current_data=current_feb11_03,
        column_mapping=column_mapping_drift,
    )

    # Convert the JSON string to a Python dictionary for pretty printing
    report_data_07 = json.loads(data_drift_report_07.json())

    # Save the report in JSON format with indentation for better readability
    with open('drift_exam_07_data_drift_feb_03.json', 'w') as f:
        json.dump(report_data_07, f, indent=4)

    # save HTML
    data_drift_report_07.save_html("drift_exam_07_data_drift_feb_03.html")

    # # Generate the regression performance report
    # regression_report = generate_regression_report(reference_data, current_data)

    # Add report to workspace
    add_report_to_workspace(workspace, PROJECT_NAME, PROJECT_DESCRIPTION, data_drift_report_07)


    # Visualize the report in evidently UI at the URL http://localhost:8000
    # evidently ui --workspace ./datascientest-workspace/