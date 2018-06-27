# Image Detection using CNTK

The objective is to use CNTK to detect custom objects in a given image. The model is trained on Azure cloud using Batch AI.

# Azure Batch AI
## Create a cluster
```
az batchai cluster create -g mahadev-ml -w mahadev-cntk -n cntk-cluster -i UbuntuDSVM -s Standard_D11 --vm-priority lowpriority --min 0 --max 1
```
## List running cluster
```
az batchai cluster list -g mahadev-ml -w mahadev-cntk
```
## Upload training script
Before you start uploading to Azure file share or blob, export following environment variables.
```
export AZURE_STORAGE_ACCOUNT=<storage account name>
export AZURE_STORAGE_KEY=<storage account key>
```
```
az storage file upload-batch --account-name mahadevmlstorage --destination scripts/detection/FastRCNN --source ./FastRCNN
```
## Upload training data
```
az storage blob upload-batch --destination data --source Grocery/
```
## Create training job
```
az batchai job create -g mahadev-ml -w mahadev-cntk -e cntk-experiment -n cntk-job-new -c cntk-cluster -f job.json 
```
## List running job
```
az batchai job list -g mahadev-ml  -e cntk-experiment -w mahadev-cntk
```
## Steam log from training job
```
az batchai job file stream --file-name stderr.txt -g mahadev-ml -w mahadev-cntk -e cntk-experiment -j cntk-job-new
```

## Delete cluster
Make sure the cluster is deleted once training is done to avoid unnecessary cloud charges.
```
az batchai cluster delete -g mahadev-ml -w mahadev-cntk --name cntk-cluster
```
In case, the training job fails or is unresponsive, delete it as follows.
```
az batchai job delete -g mahadev-ml  -e cntk-experiment -w mahadev-cntk --name cntk-job-new
```
# Issues
- The training on laptop using CPU is challenging. The training process is slow and demands significant amount of memory. I kept on getting my process killed due to memory overrun.
