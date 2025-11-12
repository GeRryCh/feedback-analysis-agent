# Deployment Guide: Streamlit App to Google Cloud Run

This guide walks you through deploying your Feedback Analysis Agent to Google Cloud Run.

## Prerequisites

1. **Google Cloud Account**
   - Create a GCP account at https://cloud.google.com
   - Create a new project or select an existing one
   - Enable billing for the project

2. **Google Cloud CLI (gcloud)**
   - Install from: https://cloud.google.com/sdk/docs/install
   - Version required: Latest stable release

3. **Docker** (for local testing)
   - Install from: https://docs.docker.com/get-docker/
   - Optional but recommended for testing before deployment

4. **OpenAI API Key**
   - Your existing key from .streamlit/secrets.toml file

## Step 1: Install and Configure gcloud CLI

### Install gcloud
```bash
# macOS (using Homebrew)
brew install --cask google-cloud-sdk

# Or download installer from:
# https://cloud.google.com/sdk/docs/install
```

### Initialize and authenticate
```bash
# Initialize gcloud and login
gcloud init

# Follow the prompts to:
# - Log in to your Google account
# - Select or create a GCP project
# - Set default region (recommend: us-central1)
```

### Set your project
```bash
# Replace PROJECT_ID with your actual GCP project ID
export PROJECT_ID="your-project-id"
gcloud config set project $PROJECT_ID
```

## Step 2: Enable Required APIs

```bash
# Enable Cloud Run API
gcloud services enable run.googleapis.com

# Enable Container Registry API
gcloud services enable containerregistry.googleapis.com

# Enable Cloud Build API (for building containers)
gcloud services enable cloudbuild.googleapis.com

# Enable Artifact Registry API (recommended for new projects)
gcloud services enable artifactregistry.googleapis.com
```

## Step 3: Set Environment Variables

```bash
# Set your project ID
export PROJECT_ID="your-project-id"

# Set your OpenAI API key (from your .streamlit/secrets.toml file)
export OPENAI_API_KEY="sk-..."

# Set app name and region
export APP_NAME="feedback-analysis-agent"
export REGION="us-central1"
```

## Step 4: Build and Deploy (Quick Method)

### Option A: Deploy directly with Cloud Build (Recommended)

```bash
# Deploy in one command - Cloud Build will build and deploy
# Note: This deploys WITHOUT authentication (good for testing/demos)
gcloud run deploy $APP_NAME \
  --source . \
  --region $REGION \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --timeout 300 \
  --set-env-vars OPENAI_API_KEY=$OPENAI_API_KEY
```

This command will:
- Build your Docker image using Cloud Build
- Push it to Artifact Registry
- Deploy to Cloud Run
- Configure all settings
- **No authentication required** (anyone with URL can access)

**For production deployments with authentication**, see the "Security Considerations > Authentication" section below.

### Option B: Build locally and push (Advanced)

```bash
# 1. Build the Docker image
docker build -t gcr.io/$PROJECT_ID/$APP_NAME:latest .

# 2. Configure Docker to use gcloud credentials
gcloud auth configure-docker

# 3. Push to Google Container Registry
docker push gcr.io/$PROJECT_ID/$APP_NAME:latest

# 4. Deploy to Cloud Run
gcloud run deploy $APP_NAME \
  --image gcr.io/$PROJECT_ID/$APP_NAME:latest \
  --region $REGION \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --timeout 300 \
  --set-env-vars OPENAI_API_KEY=$OPENAI_API_KEY
```

## Step 5: Access Your App

After deployment, gcloud will output a URL like:
```
Service URL: https://feedback-analysis-agent-xxxxx-uc.a.run.app
```

Open this URL in your browser to access your Streamlit app!

## Step 6: Verify Deployment

1. Open the Cloud Run URL in your browser
2. You should see the Streamlit interface
3. Try asking a question like: "How many feedbacks per ServiceName?"
4. Check that the app responds correctly

## Configuration Details

### Memory and CPU
- **Memory**: 2Gi (recommended for pandas + ML workloads)
- **CPU**: 1 (default, auto-allocated)
- **Timeout**: 300s (5 minutes for long-running analyses)

### Scaling
- **Min instances**: 0 (scales to zero when not in use)
- **Max instances**: 10 (default, adjust if needed)
- **Concurrency**: 80 (requests per container)

### Modify settings
```bash
# Increase memory if needed
gcloud run services update $APP_NAME \
  --region $REGION \
  --memory 4Gi

# Set minimum instances (keeps container warm)
gcloud run services update $APP_NAME \
  --region $REGION \
  --min-instances 1

# Update timeout
gcloud run services update $APP_NAME \
  --region $REGION \
  --timeout 600
```

## Updating the App

When you make changes to your code:

```bash
# Option 1: Quick redeploy (recommended)
gcloud run deploy $APP_NAME \
  --source . \
  --region $REGION

# Option 2: If using local builds
docker build -t gcr.io/$PROJECT_ID/$APP_NAME:latest .
docker push gcr.io/$PROJECT_ID/$APP_NAME:latest
gcloud run deploy $APP_NAME \
  --image gcr.io/$PROJECT_ID/$APP_NAME:latest \
  --region $REGION
```

## Updating feedback.csv

Since feedback.csv is bundled in the Docker image:

1. Replace feedback.csv with your new file
2. Redeploy using the commands above
3. The new data will be available after deployment completes

## Managing Secrets (Best Practice)

Instead of passing OPENAI_API_KEY as an environment variable, use Secret Manager:

```bash
# 1. Enable Secret Manager API
gcloud services enable secretmanager.googleapis.com

# 2. Create a secret
echo -n "$OPENAI_API_KEY" | gcloud secrets create openai-api-key \
  --data-file=-

# 3. Deploy with secret
gcloud run deploy $APP_NAME \
  --source . \
  --region $REGION \
  --set-secrets OPENAI_API_KEY=openai-api-key:latest
```

## Viewing Logs

```bash
# View recent logs
gcloud run services logs read $APP_NAME --region $REGION

# Stream logs in real-time
gcloud run services logs tail $APP_NAME --region $REGION
```

## Cost Optimization

### Cloud Run Costs
- **Free tier**: 2 million requests/month, 360,000 GB-seconds
- **Pricing**: Pay only when requests are being handled
- **Optimization**: Scales to zero when not in use (no cost)

### OpenAI API Costs
- Each query makes 2-3 GPT-4o API calls
- Monitor usage at: https://platform.openai.com/usage
- Consider rate limiting for production use

### Estimated costs
- Light usage (100 queries/day): ~$5-10/month (mostly OpenAI)
- Cloud Run: ~$1-3/month
- OpenAI: ~$4-7/month

## Troubleshooting

### Issue: "Permission denied" errors
```bash
# Grant yourself necessary roles
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="user:your-email@example.com" \
  --role="roles/run.admin"
```

### Issue: Build fails
```bash
# Check Docker build locally first
docker build -t test-image .

# Check logs
gcloud builds log --limit 5
```

### Issue: App crashes on startup
```bash
# Check logs for errors
gcloud run services logs read $APP_NAME --region $REGION --limit 50

# Common issues:
# - Missing OPENAI_API_KEY
# - feedback.csv not found
# - Port configuration incorrect
```

### Issue: Timeout errors
```bash
# Increase timeout
gcloud run services update $APP_NAME \
  --region $REGION \
  --timeout 600

# Increase memory
gcloud run services update $APP_NAME \
  --region $REGION \
  --memory 4Gi
```

### Issue: "File not found: feedback.csv"
- Ensure feedback.csv is in the root directory
- Check .dockerignore doesn't exclude it
- Rebuild and redeploy

## Security Considerations

### Authentication

The app includes **optional username/password authentication** using Streamlit's secrets management.

**By default, authentication is DISABLED** to allow easy local development. You only need to enable it for cloud deployments.

#### Local Development (No Authentication Required)

For local development, simply configure your secrets file:

```bash
# Create .streamlit directory if it doesn't exist
mkdir -p .streamlit

# Set your OpenAI API key in secrets.toml
cat > .streamlit/secrets.toml << 'EOF'
OPENAI_API_KEY = "sk-your-key-here"
REQUIRE_AUTH = false
EOF

# Run the app - no authentication required
streamlit run app.py
```

The app will work immediately with your own feedback.csv file and API key, no passwords needed.

#### Enabling Authentication (Optional - For Cloud Deployments)

If you want to add authentication (recommended for cloud deployments), set `REQUIRE_AUTH=true`:

**Option 1: Using `.streamlit/secrets.toml` (Local Testing)**
```toml
OPENAI_API_KEY = "sk-your-key-here"

# Enable authentication
REQUIRE_AUTH = true

[passwords]
admin = "secure_password_123"
demo_user = "demo_password_456"
```

**Option 2: Using Environment Variable**
```bash
export REQUIRE_AUTH=true
streamlit run app.py
```

#### Cloud Run Deployment with Authentication

**Deploying WITH Authentication (Recommended for Production)**

```bash
# 1. Enable Secret Manager API
gcloud services enable secretmanager.googleapis.com

# 2. Create secret for OpenAI API key
echo -n "$OPENAI_API_KEY" | gcloud secrets create openai-api-key \
  --data-file=-

# 3. Create a secrets.toml file with authentication enabled
cat > cloud_secrets.toml << 'EOF'
OPENAI_API_KEY = "placeholder"  # This will be overridden by secret manager
REQUIRE_AUTH = true

[passwords]
admin = "your_secure_admin_password_here"
demo_user = "your_demo_password_here"
analyst = "another_secure_password"
EOF

# 4. Create secret from file
gcloud secrets create streamlit-secrets \
  --data-file=cloud_secrets.toml

# 5. Clean up local file (important!)
rm cloud_secrets.toml

# 6. Deploy with authentication enabled
gcloud run deploy $APP_NAME \
  --source . \
  --region $REGION \
  --allow-unauthenticated \
  --memory 2Gi \
  --timeout 300 \
  --set-secrets OPENAI_API_KEY=openai-api-key:latest \
  --update-secrets /app/.streamlit/secrets.toml=streamlit-secrets:latest
```

**Deploying WITHOUT Authentication (For Testing/Public Demos)**

```bash
# Simple deployment without authentication
# Users can access immediately without login
gcloud run deploy $APP_NAME \
  --source . \
  --region $REGION \
  --allow-unauthenticated \
  --memory 2Gi \
  --timeout 300 \
  --set-env-vars OPENAI_API_KEY=$OPENAI_API_KEY
```

**Note**:
- `REQUIRE_AUTH` defaults to `false` if not set
- With authentication disabled, anyone with the URL can access the app
- With authentication enabled, users must log in with username/password

#### Managing User Credentials (When Authentication is Enabled)

**Add a new user:**
```bash
# 1. Get current secret
gcloud secrets versions access latest --secret=streamlit-secrets > temp_secrets.toml

# 2. Edit file to add new user in [passwords] section
nano temp_secrets.toml  # or vim, code, etc.
# Add line: new_user = "secure_password_123"

# 3. Create new version
gcloud secrets versions add streamlit-secrets \
  --data-file=temp_secrets.toml

# 4. Clean up
rm temp_secrets.toml

# 5. Redeploy to pick up new secret version
gcloud run deploy $APP_NAME --source . --region $REGION
```

**Remove a user:**
```bash
# 1. Get current secret
gcloud secrets versions access latest --secret=streamlit-secrets > temp_secrets.toml

# 2. Edit file to remove user from [passwords] section
nano temp_secrets.toml  # or vim, code, etc.

# 3. Create new version
gcloud secrets versions add streamlit-secrets \
  --data-file=temp_secrets.toml

# 4. Clean up
rm temp_secrets.toml

# 5. Redeploy
gcloud run deploy $APP_NAME --source . --region $REGION
```

**View current users:**
```bash
# View the secrets file (will show usernames and passwords)
gcloud secrets versions access latest --secret=streamlit-secrets

# Or extract just the [passwords] section
gcloud secrets versions access latest --secret=streamlit-secrets | \
  sed -n '/\[passwords\]/,/^$/p'
```

#### Additional Cloud Run IAM Authentication (Optional Layer)

For additional security, you can also require Cloud Run IAM authentication:

```bash
# Remove unauthenticated access
gcloud run services remove-iam-policy-binding $APP_NAME \
  --region $REGION \
  --member="allUsers" \
  --role="roles/run.invoker"

# Add specific Google accounts
gcloud run services add-iam-policy-binding $APP_NAME \
  --region $REGION \
  --member="user:email@example.com" \
  --role="roles/run.invoker"
```

With this setup, users must:
1. First authenticate via Google (Cloud Run IAM)
2. Then log in with username/password (Streamlit app)

### API Key Security
- Use Secret Manager instead of environment variables
- Rotate keys regularly
- Monitor API usage for anomalies

### Code Execution Warning
The app uses `allow_dangerous_code=True` for the pandas agent, which executes AI-generated code. This is contained within the Cloud Run sandbox, but be aware of this security consideration.

## Cleanup

To delete all resources:

```bash
# Delete Cloud Run service
gcloud run services delete $APP_NAME --region $REGION

# Delete container images
gcloud container images delete gcr.io/$PROJECT_ID/$APP_NAME:latest

# Delete secrets (if created)
gcloud secrets delete openai-api-key
```

## Next Steps

1. Set up custom domain (optional)
2. Configure Cloud Monitoring alerts
3. Set up Cloud Logging exports
4. Implement authentication
5. Add rate limiting
6. Set up CI/CD with Cloud Build triggers

## Support Resources

- Cloud Run documentation: https://cloud.google.com/run/docs
- Streamlit on Cloud Run: https://docs.streamlit.io/deploy/streamlit-community-cloud
- GCP pricing calculator: https://cloud.google.com/products/calculator

## Quick Reference

```bash
# Deploy
gcloud run deploy $APP_NAME --source . --region $REGION

# View logs
gcloud run services logs tail $APP_NAME --region $REGION

# Get URL
gcloud run services describe $APP_NAME --region $REGION --format 'value(status.url)'

# Delete service
gcloud run services delete $APP_NAME --region $REGION
```
