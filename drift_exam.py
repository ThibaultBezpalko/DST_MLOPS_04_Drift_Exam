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
from evidently.metric_preset import DataDriftPreset, RegressionPreset
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
## ANALYSE DERIVE - 01
############################

# # Train the production model
# regressor.fit(reference_jan11[numerical_features + categorical_features], reference_jan11[target])

# # Perform column mapping
# column_mapping = ColumnMapping()
# column_mapping.target = target
# column_mapping.prediction = prediction
# column_mapping.numerical_features = numerical_features
# column_mapping.categorical_features = categorical_features

# # Generate predictions for the reference data
# ref_prediction = regressor.predict(reference_jan11[numerical_features + categorical_features])
# reference_jan11['prediction'] = ref_prediction

# # Initialize the regression performance report with the default regression metrics preset
# regression_report = Report(metrics=[
#     RegressionPreset(),
# ])

# # Run the regression performance report using the reference data
# regression_report.run(reference_data=None, 
#                                   current_data=reference_jan11,
#                                   column_mapping=column_mapping)


############################
## ANALYSE DERIVE - 02
############################

# column_mapping_drift = ColumnMapping()

# column_mapping_drift.target = target
# column_mapping_drift.prediction = prediction
# column_mapping_drift.numerical_features = numerical_features
# column_mapping_drift.categorical_features = []

# data_drift_report = Report(metrics=[
#     DataDriftPreset(),
# ])

# data_drift_report.run(
#     reference_data=reference_jan11,
#     current_data=current_feb11.loc['2011-02-14 00:00:00':'2011-02-21 23:00:00'],
#     column_mapping=column_mapping_drift,
# )



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

if __name__ == "__main__":
    WORKSPACE_NAME = "datascientest-workspace"
    PROJECT_NAME = "drift_monitoring_exam"
    PROJECT_DESCRIPTION = "Evidently Dashboards"

    # load data
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

    # Train test split ONLY on reference_jan11
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        reference_jan11[numerical_features + categorical_features],
        reference_jan11[target],
        test_size=0.3
    )

    # Model training
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
    column_mapping = ColumnMapping()

    # Map the actual target and prediction column names in the dataset for evidently
    column_mapping.target = 'target'
    column_mapping.prediction = 'prediction'

    # Specify which features are numerical and which are categorical for the evidently report
    column_mapping.numerical_features = numerical_features
    column_mapping.categorical_features = categorical_features

    # Initialize the regression performance report with the default regression metrics preset
    regression_report = Report(metrics=[
        RegressionPreset(),
    ])

    # Run the regression performance report using the training data as reference and test data as current
    # The data is sorted by index to ensure consistent ordering for the comparison
    regression_report.run(reference_data=X_train.sort_index(), 
                                    current_data=X_test.sort_index(),
                                    column_mapping=column_mapping)

    # Convert the JSON string to a Python dictionary for pretty printing
    report_data = json.loads(regression_report.json())

    # Save the report in JSON format with indentation for better readability
    with open('data_drift_report.json', 'w') as f:
        json.dump(report_data, f, indent=4)

    # save HTML
    regression_report.save_html("Classification Report.html")

    # # Generate the regression performance report
    # regression_report = generate_regression_report(reference_data, current_data)

    # Create and Add report to workspace
    workspace = Workspace.create(WORKSPACE_NAME)
    add_report_to_workspace(workspace, PROJECT_NAME, PROJECT_DESCRIPTION, regression_report)

    # Visualize the report in evidently UI at the URL http://localhost:8000
    # evidently ui --workspace ./datascientest-workspace/