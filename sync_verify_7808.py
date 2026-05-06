from apps.runs.models import Run
run = Run.objects.get(run_id='run-7808c90045e9')
print('RUN', run.state, run.selected_slide_count, run.label_counts, run.feature_extractor_used)
for link in run.approach_links.select_related('approach_template').all().order_by('approach_template__label'):
    print('LINK', link.approach_template.label, link.state, link.mean_auroc, link.mean_f1_macro, link.prediction_artifacts)
