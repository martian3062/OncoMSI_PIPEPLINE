from django.urls import path

from .views import results_beta_infer, results_beta_page


urlpatterns = [
    path("", results_beta_page, name="dashboard"),
    path("results/", results_beta_page, name="results-page"),
    path("results-beta/", results_beta_page, name="results-beta-page"),
    path("results-beta/infer/", results_beta_infer, name="results-beta-infer"),
]
