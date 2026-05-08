from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("runs", "0003_run_external_cohorts_and_runapproachlink_richer_metrics"),
    ]

    operations = [
        migrations.AddField(
            model_name="run",
            name="repeat_seeds",
            field=models.JSONField(blank=True, default=list),
        ),
    ]
