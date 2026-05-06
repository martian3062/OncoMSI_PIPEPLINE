# Generated manually on 2026-05-06 because local Django is not installed.

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("runs", "0002_run_bundle_config_path_run_remote_launch_log_path_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="run",
            name="external_cohorts",
            field=models.JSONField(blank=True, default=list),
        ),
        migrations.AddField(
            model_name="runapproachlink",
            name="aggregate_confusion_matrix",
            field=models.JSONField(blank=True, default=dict),
        ),
        migrations.AddField(
            model_name="runapproachlink",
            name="auroc_ci_high",
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="runapproachlink",
            name="auroc_ci_low",
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="runapproachlink",
            name="auroc_per_fold",
            field=models.JSONField(blank=True, default=list),
        ),
        migrations.AddField(
            model_name="runapproachlink",
            name="auroc_std",
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="runapproachlink",
            name="available_bag_slide_count",
            field=models.PositiveIntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="runapproachlink",
            name="external_metrics",
            field=models.JSONField(blank=True, default=dict),
        ),
        migrations.AddField(
            model_name="runapproachlink",
            name="fold_metrics",
            field=models.JSONField(blank=True, default=list),
        ),
        migrations.AddField(
            model_name="runapproachlink",
            name="mean_auprc",
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="runapproachlink",
            name="mean_balanced_accuracy",
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="runapproachlink",
            name="mean_best_threshold",
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="runapproachlink",
            name="mean_brier_score",
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="runapproachlink",
            name="mean_f1_macro_default_threshold",
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="runapproachlink",
            name="mean_precision",
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="runapproachlink",
            name="mean_recall_msi_h",
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="runapproachlink",
            name="mean_specificity",
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="runapproachlink",
            name="missing_bag_slides",
            field=models.JSONField(blank=True, default=list),
        ),
    ]
