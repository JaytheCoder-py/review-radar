# Deploying Review Radar

Two Cloud Run deployments from one image: a **job** that runs nightly and exits, and a
**service** that serves the dashboard. Nothing here needs a key file — Vertex is reached
through the job's own service-account identity (D-005).

Set once:

```bash
export PROJECT=your-gcp-project REGION=us-central1 VERTEX_REGION=us-east5
```

## 1. Enable the APIs

```bash
gcloud services enable run.googleapis.com cloudscheduler.googleapis.com aiplatform.googleapis.com artifactregistry.googleapis.com --project "$PROJECT"
```

## 2. A service account with only what it needs

`roles/aiplatform.user` is the whole grant. The job reads from the SEC over the public
internet and writes to a mounted volume; it needs no storage-admin, no project-editor,
and nothing that could touch another service.

```bash
gcloud iam service-accounts create reviewradar --display-name="Review Radar" --project "$PROJECT"
```

```bash
gcloud projects add-iam-policy-binding "$PROJECT" --member="serviceAccount:reviewradar@$PROJECT.iam.gserviceaccount.com" --role="roles/aiplatform.user"
```

## 3. Build and push

```bash
gcloud artifacts repositories create reviewradar --repository-format=docker --location="$REGION" --project "$PROJECT"
```

```bash
gcloud builds submit --tag "$REGION-docker.pkg.dev/$PROJECT/reviewradar/job:latest" --project "$PROJECT"
```

## 4. The nightly job

`--task-timeout` is generous because a day with 400 filings at 8 requests/second against
EDGAR takes about a minute of wall clock before any extraction happens.

```bash
gcloud run jobs create reviewradar-nightly --image "$REGION-docker.pkg.dev/$PROJECT/reviewradar/job:latest" --region "$REGION" --service-account "reviewradar@$PROJECT.iam.gserviceaccount.com" --set-env-vars "GOOGLE_CLOUD_PROJECT=$PROJECT,VERTEX_REGION=$VERTEX_REGION" --task-timeout=30m --max-retries=1 --args="ingest,--date,TODAY,--contact,you@example.com,--vertex,--project,$PROJECT" --project "$PROJECT"
```

## 5. The schedule

EDGAR's daily index for a given day publishes late that evening US Eastern. Running at
06:00 Eastern the next morning reads a complete index rather than a partial one — a job
that runs too early silently ingests a fraction of the day and marks it done.

```bash
gcloud scheduler jobs create http reviewradar-trigger --schedule="0 6 * * 2-6" --time-zone="America/New_York" --uri="https://$REGION-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/$PROJECT/jobs/reviewradar-nightly:run" --http-method=POST --oauth-service-account-email="reviewradar@$PROJECT.iam.gserviceaccount.com" --project "$PROJECT"
```

`2-6` is Tuesday to Saturday: it processes Monday through Friday's filings, one day in
arrears. Weekends have no daily index and the job would record a failure for a date the
SEC never published.

### Forward verification, weekly

`reviewradar verify` checks past-dated extractions against the tape and appends verdicts.
Weekly rather than nightly, and Sunday rather than a weekday: an ex-date announced today is
verifiable in a few days' time, not tonight, so a nightly run would spend most of its
output re-recording `unverifiable — the ex-date has not passed`. The verdict log is
append-only and idempotent on the verdict's own content, so re-running costs nothing and a
verdict that changes when an ex-date finally passes appends beside the old one.

```bash
gcloud run jobs create reviewradar-verify --image "$REGION-docker.pkg.dev/$PROJECT/reviewradar/job:latest" --region "$REGION" --service-account "reviewradar@$PROJECT.iam.gserviceaccount.com" --task-timeout=15m --max-retries=1 --args="verify,--db,/data/events.duckdb,--source,yahoo" --project "$PROJECT"
```

```bash
gcloud scheduler jobs create http reviewradar-verify-trigger --schedule="0 8 * * 0" --time-zone="America/New_York" --uri="https://$REGION-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/$PROJECT/jobs/reviewradar-verify:run" --http-method=POST --oauth-service-account-email="reviewradar@$PROJECT.iam.gserviceaccount.com" --project "$PROJECT"
```

`--source yahoo` reaches a public endpoint with no key and no account, so the job needs no
extra IAM grant. **It must stay `yahoo` or another source that serves unadjusted, as-traded
closes.** A back-adjusted series has the split divided out of every price before the
ex-date, so the step this job exists to look for is not in the data — and the failure is
silent, because the output is still full of confident-looking verdicts. See D-008.

The signal to watch is not the exit code. A run that verifies nothing exits zero; check
that `contradicted` is being read by somebody, because it is the only output here that asks
for a human.

## 6. The dashboard

```bash
gcloud run deploy reviewradar-web --image "$REGION-docker.pkg.dev/$PROJECT/reviewradar/job:latest" --region "$REGION" --service-account "reviewradar@$PROJECT.iam.gserviceaccount.com" --allow-unauthenticated --port 8080 --command sh --args="-c,reviewradar serve --db /data/events.duckdb --port 8080" --project "$PROJECT"
```

Public and read-only by design: there is nothing here that is not already a public SEC
filing, and requiring a login on a page whose entire content is public would be theatre.

## Verifying it actually ran

A job that exits zero having ingested nothing looks identical to a healthy one on the
Cloud Run dashboard. Check the count, not the status:

```bash
gcloud run jobs executions list --job reviewradar-nightly --region "$REGION" --project "$PROJECT" --limit 5
```

Then hit `/healthz`, which reports the number of events in the log. A flat count across
several days is the signal that something is wrong, and it is the one the Cloud Run
console will not give you.

## Cost

The nightly job processes roughly 200–400 8-K filings. The baseline eliminates about 42%
before any model call, so 120–230 filings reach Vertex. At Haiku pricing and roughly 6k
input tokens per filing, that is a few US cents per night. Cloud Run's free tier covers
both the job and the service at this volume.
