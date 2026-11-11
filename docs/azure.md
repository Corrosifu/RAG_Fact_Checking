# Deploying to Azure App Service

Multi-container deployment using `docker-compose.prod.yml`. The backend image is built from
`docker/api.Dockerfile`, the frontend from `docker/ui.Dockerfile`.

> Build the index before deploying (`ragfc sample && ragfc index`, or `ragfc ingest && ragfc index`)
> and make `data/` available to the backend container (e.g. an Azure Files share mounted on
> `/app/data`). The images do not contain the index.
> To use neural models, build the backend with `--build-arg EXTRAS=api,ml` and set
> `RAGFC_EMBEDDER=sentence-transformers` / `RAGFC_GENERATOR=hf` in the app settings.

After login:

```powershell
$env:HF_TOKEN = "hf_xxx_your_token_here"

$RG  = "rg-rag-portfolio"
$LOC = "francecentral"
$APP = "rag-portfolio-app"
$ACR = "ragportfolioacr"   # pick a unique name

az group create -n $RG -l $LOC

az acr create -n $ACR -g $RG --sku Standard
$REG_LOGIN_SERVER = az acr show -n $ACR -g $RG --query loginServer -o tsv

az acr build -r $ACR -t "$REG_LOGIN_SERVER/rag-backend:latest"  -f docker/api.Dockerfile --build-arg EXTRAS=api .
az acr build -r $ACR -t "$REG_LOGIN_SERVER/rag-frontend:latest" -f docker/ui.Dockerfile .

az appservice plan create -g $RG -n "asp-rag" --is-linux --sku P1v3
az webapp create -g $RG -p "asp-rag" -n $APP `
  --multicontainer-config-type compose `
  --multicontainer-config-file docker-compose.prod.yml

az webapp identity assign -g $RG -n $APP
$PRINCIPAL_ID = az webapp identity show -g $RG -n $APP --query principalId -o tsv
$ACR_ID       = az acr show -n $ACR -g $RG --query id -o tsv
az role assignment create --assignee $PRINCIPAL_ID --role "AcrPull" --scope $ACR_ID

$STG = "stragcache$([System.Guid]::NewGuid().ToString('N').Substring(0,8))"
az storage account create -g $RG -n $STG -l $LOC --sku Standard_LRS
az storage share-rm create -g $RG --storage-account $STG --name hf-cache
$STG_KEY = az storage account keys list -g $RG -n $STG --query "[0].value" -o tsv

az webapp config storage-account add -g $RG -n $APP `
  --custom-id hf-cache --storage-type AzureFiles `
  --account-name $STG --share-name hf-cache --access-key "$STG_KEY" `
  --mount-path /hf_cache

az webapp config appsettings set -g $RG -n $APP --settings `
  REG_LOGIN_SERVER=$REG_LOGIN_SERVER `
  HF_TOKEN="$env:HF_TOKEN"

az webapp config container set -g $RG -n $APP `
  --multicontainer-config-type compose `
  --multicontainer-config-file docker-compose.prod.yml
```

