# Replace with your details
GITHUB_TOKEN="$1"
REPO="$2"
COMMIT_SHA="$3"

# Step 1: Get the workflow run ID for the specified commit
RUN_ID=$(curl -s -H "Authorization: Bearer $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  "https://api.github.com/repos/$REPO/actions/runs?head_sha=$COMMIT_SHA" \
  | jq -r '.workflow_runs[1].id')

# Check if RUN_ID was found
if [ -z "$RUN_ID" ]; then
  echo "❌ No workflow run found for the commit."
  exit 1
fi

echo "Workflow Run ID: $RUN_ID"

# Step 2: Fetch job statuses for the specific workflow run
JOB_STATUSES=$(curl -s -H "Authorization: Bearer $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  "https://api.github.com/repos/$REPO/actions/runs/$RUN_ID/jobs" \
  | jq -r '.jobs[] | select(.name == "integration-tests" or .name == "pre-commit") | {name: .name, status: .status, conclusion: .conclusion}')

# Output the job statuses for debugging
echo "Job statuses found: $JOB_STATUSES"
if [ -z "$JOB_STATUSES" ]; then
  echo "❌ No Jobs found for the workflow."
  exit 1
fi

# Step 3: Check if all jobs have completed successfully
ALL_SUCCESS=1
while read -r job; do
  JOB_NAME=$(echo "$job" | jq -r '.name')
  JOB_STATUS=$(echo "$job" | jq -r '.status')
  JOB_CONCLUSION=$(echo "$job" | jq -r '.conclusion' | xargs)
  
  echo "Job: $JOB_NAME, Status: $JOB_STATUS, Conclusion: $JOB_CONCLUSION"
  
  if [[ "$JOB_CONCLUSION" != "success" ]]; then
    ALL_SUCCESS=0
  fi
done < <(echo "$JOB_STATUSES" | jq -c '.')

if [ $ALL_SUCCESS -ne 1 ]; then
  echo "❌ One or more specified jobs did not pass."
  exit 1
else
  echo "✅ All specified jobs passed successfully."
fi
