from django.urls import path

from .views import (
    analysis_summary_api,
    batch_status_api,
    frontend_redirect,
    predict_job_create_api,
    prediction_history_api,
    predict_job_status_api,
    predict_metadata_api,
    predict_upload_api,
    retired_frontend_partial,
    storage_sample_test_api,
    storage_samples_api,
)


urlpatterns = [
    path("api/predict-metadata/", predict_metadata_api, name="predict-metadata-api"),
    path("api/predict-jobs/", predict_job_create_api, name="predict-job-create-api"),
    path("api/predict-jobs/<str:job_id>/", predict_job_status_api, name="predict-job-status-api"),
    path("api/predict-upload/", predict_upload_api, name="predict-upload-api"),
    path("api/storage-samples/", storage_samples_api, name="storage-samples-api"),
    path("api/storage-samples/test/", storage_sample_test_api, name="storage-samples-test-api"),
    path("api/batch-status/", batch_status_api, name="batch-status-api"),
    path("api/analysis-summary/", analysis_summary_api, name="analysis-summary-api"),
    path("api/prediction-history/", prediction_history_api, name="prediction-history-api"),
    path("", frontend_redirect, name="dashboard"),
    path("results/", frontend_redirect, name="results-page"),
    path("results-beta/", frontend_redirect, name="results-beta-page"),
    path("partials/live-runs/", retired_frontend_partial, name="live-runs-partial-retired"),
    path("partials/milestones/", retired_frontend_partial, name="milestones-partial-retired"),
]
