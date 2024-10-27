#! /usr/bin/sh

# project name
project="titanic"

# Create random string
guid=$(cat /proc/sys/kernel/random/uuid)
suffix=${guid//[-]/}
suffix=${suffix:0:18}

# Set the necessary variables
RESOURCE_GROUP="rg-${suffix}"
RESOURCE_PROVIDER="Microsoft.MachineLearning"
REGION="westus"
WORKSPACE_NAME="mlw-${suffix}"
COMPUTE_INSTANCE="ci-${suffix}"
COMPUTE_SIZE="STANDARD_A1_V2"
COMPUTE_CLUSTER="aml-cluster"

# Register the Azure Machine Learning resource provider in the subscription
echo "Register the Machine Learning resource provider:"
az provider register --namespace $RESOURCE_PROVIDER

# Create the resource group and workspace and set to default
echo "Create a resource group and set as default:"
az group create --name $RESOURCE_GROUP --location $REGION
az configure --defaults group=$RESOURCE_GROUP

echo "Create an Azure Machine Learning workspace:"
az ml workspace create --name $WORKSPACE_NAME 
az configure --defaults workspace=$WORKSPACE_NAME 

# Create compute instance
echo "Creating a compute instance with name: " $COMPUTE_INSTANCE
az ml compute create --name ${COMPUTE_INSTANCE} --size ${COMPUTE_SIZE} --type ComputeInstance 

# Create compute cluster
echo "Creating a compute cluster with name: " $COMPUTE_CLUSTER
az ml compute create --name ${COMPUTE_CLUSTER} --size ${COMPUTE_SIZE} --max-instances 2 --type AmlCompute 

# Create a file data asset
echo "Creating a data asset with name: " ${project}-local
az ml data create --name "${project}-local" --type uri_file --description Data asset pointing to a local file --path ./data/titanic.csv 

# Create a file data asset
echo "Creating a data asset with name: " ${project}-sample-local
az ml data create --name "${project}-sample-local" --type uri_file --description Data asset pointing to a local file --path ./data/titanic_sample.csv

# Create components
echo "Creating components"
az ml component create --file ./prep_data.yml 
az ml component create --file ./train_model.yml 
az ml component create --file ./make_predictions.yml 
