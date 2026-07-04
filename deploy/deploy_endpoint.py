import boto3
import sagemaker
import os
import json
import time
from sagemaker.sklearn.model import SKLearnModel


def deploy_model():

    # ─── Setup ──────────────────────────────────────────────
    region  = os.environ.get("AWS_REGION", "us-east-1")
    account = boto3.client("sts", region_name=region)\
                   .get_caller_identity()["Account"]
    bucket  = f"sagemaker-adult-income-cdk-{account}"

    sm_client = boto3.client("sagemaker",         region_name=region)
    s3_client = boto3.client("s3",                region_name=region)
    runtime   = boto3.client("sagemaker-runtime", region_name=region)

    print(f"✅ Region  : {region}")
    print(f"✅ Account : {account}")

    # ─── Read Infra Outputs from S3 ─────────────────────────
    try:
        response  = s3_client.get_object(
            Bucket=bucket,
            Key="infra/infra_outputs.json"
        )
        infra     = json.loads(response["Body"].read())
        role      = infra["role_arn"]
        domain_id = infra["domain_id"]
        bucket    = infra["bucket_name"]

        print(f"✅ Infra outputs loaded")
        print(f"✅ Domain ID : {domain_id}")
        print(f"✅ Role ARN  : {role}")
        print(f"✅ Bucket    : {bucket}")

    except Exception as e:
        print(f"⚠️ S3 read failed : {str(e)}")
        role = os.environ.get("AWS_ROLE_ARN")
        print(f"✅ Fallback Role  : {role}")

    # ─── Get Latest Approved Model ──────────────────────────
    try:
        model_packages = sm_client.list_model_packages(
            ModelPackageGroupName="AdultIncomeGroup",
            ModelApprovalStatus="Approved",
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=1
        )

        if not model_packages["ModelPackageSummaryList"]:
            raise Exception(
                "No approved model found in AdultIncomeGroup! "
                "Please approve model first."
            )

        model_package_arn = model_packages[
            "ModelPackageSummaryList"
        ][0]["ModelPackageArn"]

        model_version = model_packages[
            "ModelPackageSummaryList"
        ][0]["ModelPackageVersion"]

        package_detail = sm_client.describe_model_package(
            ModelPackageName=model_package_arn
        )
        model_data_url = package_detail["InferenceSpecification"]\
                          ["Containers"][0]["ModelDataUrl"]

        print(f"✅ Model Package ARN     : {model_package_arn}")
        print(f"✅ Model Package Version : {model_version}")
        print(f"✅ Model Data URL        : {model_data_url}")

    except Exception as e:
        print(f"❌ Model fetch error: {str(e)}")
        raise

    # ─── Create SageMaker Session ───────────────────────────
    session = sagemaker.Session(
        boto_session=boto3.Session(region_name=region),
        sagemaker_client=sm_client,
        default_bucket=bucket
    )

    # ─── Create Model WITH Inference Entry Point ────────────
    model_name    = f"adult-income-predictor-{int(time.time())}"
    endpoint_name = "adult-income-predictor-endpoint"
    config_name   = f"adult-endpoint-config-{int(time.time())}"   # ✅ Unique each run

    try:
        # ✅ explicit entry_point sets SAGEMAKER_PROGRAM correctly
        # so the container can find model_fn/input_fn/predict_fn/output_fn
        model = SKLearnModel(
            model_data=model_data_url,
            role=role,
            entry_point="train.py",
            source_dir="pipeline",
            framework_version="1.2-1",
            sagemaker_session=session,
            name=model_name
        )
        print(f"✅ Model object created : {model_name}")

    except Exception as e:
        print(f"❌ Model creation error: {str(e)}")
        raise

    # ─── Check for Existing Endpoint (update, don't recreate) ─
    endpoint_exists = False
    try:
        endpoint_status = sm_client.describe_endpoint(
            EndpointName=endpoint_name
        )["EndpointStatus"]
        print(f"✅ Endpoint exists : {endpoint_name}")
        print(f"✅ Status          : {endpoint_status}")

        # Wait out any in-flight create/update before deciding
        while endpoint_status in ["Creating", "Updating",
                                  "SystemUpdating", "RollingBack"]:
            print(f"⏳ Endpoint busy ({endpoint_status}), waiting...")
            time.sleep(30)
            endpoint_status = sm_client.describe_endpoint(
                EndpointName=endpoint_name
            )["EndpointStatus"]

        if endpoint_status == "InService":
            # Healthy endpoint: update in place (managed blue/green,
            # zero downtime, automatic rollback on failure)
            endpoint_exists = True
            print(f"✅ Will update existing endpoint in place")
        else:
            # Failed / OutOfService endpoints cannot be updated
            print(f"⏳ Endpoint status '{endpoint_status}' cannot be "
                  f"updated, deleting before recreate...")
            sm_client.delete_endpoint(EndpointName=endpoint_name)

            while True:
                try:
                    sm_client.describe_endpoint(EndpointName=endpoint_name)
                    time.sleep(10)
                except sm_client.exceptions.ClientError:
                    break

            print(f"✅ Unhealthy endpoint deleted, will create fresh one")

    except sm_client.exceptions.ClientError:
        print(f"⏳ No existing endpoint found, will create fresh one")

    # ─── Create Model in SageMaker ──────────────────────────
    model.create(instance_type="ml.m5.large")
    print(f"✅ Model created in SageMaker")

    # ─── Create New Endpoint Config (unique name) ───────────
    sm_client.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                "VariantName"         : "AllTraffic",
                "ModelName"           : model_name,
                "InitialInstanceCount": 1,
                "InstanceType"        : "ml.m5.large",
                "InitialVariantWeight": 1
            }
        ]
    )
    print(f"✅ Endpoint config created : {config_name}")

    # ─── Create or Update Endpoint ──────────────────────────
    if endpoint_exists:
        sm_client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )
        print(f"✅ Endpoint update started (blue/green, zero downtime)!")
    else:
        sm_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )
        print(f"✅ Endpoint creation started!")

    # ─── Wait for Endpoint InService ────────────────────────
    print(f"⏳ Waiting for endpoint to be InService...")

    while True:
        detail = sm_client.describe_endpoint(EndpointName=endpoint_name)
        status = detail["EndpointStatus"]

        print(f"   Status: {status}")

        if status == "InService":
            # A failed update rolls back to the old config and returns
            # InService — verify the new config is actually live
            if detail["EndpointConfigName"] != config_name:
                raise Exception(
                    f"Endpoint update failed and rolled back to previous "
                    f"config '{detail['EndpointConfigName']}': "
                    f"{detail.get('FailureReason', 'no reason reported')}"
                )
            break
        elif status in ["Failed", "OutOfService"]:
            raise Exception(
                f"Endpoint failed: {detail.get('FailureReason', status)}"
            )

        time.sleep(30)

    print(f"✅ Endpoint is InService!")

    # ─── Clean Up Old Endpoint Configs (avoid clutter) ──────
    # Runs only after the new config is live; never deletes the
    # config currently serving traffic
    try:
        configs = sm_client.list_endpoint_configs(
            NameContains="adult-endpoint-config"
        )
        for cfg in configs.get("EndpointConfigs", []):
            if cfg["EndpointConfigName"] == config_name:
                continue
            try:
                sm_client.delete_endpoint_config(
                    EndpointConfigName=cfg["EndpointConfigName"]
                )
                print(f"✅ Deleted old config: {cfg['EndpointConfigName']}")
            except Exception:
                pass
    except Exception as e:
        print(f"⚠️ Config cleanup skipped: {str(e)}")

    # ─── Test Endpoint ───────────────────────────────────────
    print(f"\n⏳ Testing endpoint...")

    # Feature order: age, education_num, hours_per_week, capital_gain, capital_loss, sex_male, is_married
    test_cases = [
        ("39,13,40,2174,0,1,0",  "39yo, 13yr edu, 40hrs/wk, married, male"),
        ("25,9,20,0,0,0,1",      "25yo, 9yr edu, 20hrs/wk, single, female"),
        ("50,14,60,15000,0,1,1", "50yo, high edu, 60hrs/wk, married, male"),
    ]

    for test_input, description in test_cases:
        response   = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="text/csv",
            Body=test_input
        )
        prediction = response["Body"].read().decode("utf-8").strip()
        result     = ">50K ✅" if prediction == "1" else "<=50K"

        print(f"   Input      : {test_input}")
        print(f"   Scenario   : {description}")
        print(f"   Prediction : {result}")
        print()

    # ─── Save Deploy Outputs to S3 ──────────────────────────
    deploy_outputs = {
        "endpoint_name"    : endpoint_name,
        "model_name"       : model_name,
        "model_package_arn": model_package_arn,
        "model_version"    : model_version,
        "group_name"       : "AdultIncomeGroup",
        "domain_id"        : domain_id,
        "status"           : "InService",
        "region"           : region
    }

    s3_client.put_object(
        Bucket=bucket,
        Key="deploy/deploy_outputs.json",
        Body=json.dumps(deploy_outputs, indent=2)
    )

    print("\n" + "="*50)
    print("✅ DEPLOYMENT COMPLETE!")
    print(f"   Endpoint    : {endpoint_name}")
    print(f"   Model       : {model_name}")
    print(f"   Version     : {model_version}")
    print(f"   Status      : InService ✅")
    print("="*50)

    return deploy_outputs


if __name__ == "__main__":
    deploy_model()