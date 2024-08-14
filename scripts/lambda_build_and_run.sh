

#git clone https://${GH_PAT}@github.com/clorenz7/dpo.git dpo

if [ ! -d "venv_dpo" ]; then
    python3 -m venv --system-site-packages venv_dpo

    source venv_dpo/bin/activate
    pip install --upgrade setuptools wheel
    cd ~/dpo
    pip install -e . -U
else
    source venv_dpo/bin/activate
    cd ~/dpo
fi

export CDPO_DEFAULT_DIR="/home/ubuntu/cdpo-fs"

current_datetime=$(date +"%Y_%b_%d_%H_%M")

python scripts/train_and_eval.py -p $1 -o $CDPO_DEFAULT_DIR | tee $CDPO_DEFAULT_DIR/${current_datetime}.log

# LAMBDA_API_KEY=""
# INSTANCE_IDS_FILE=

# curl -u $LAMBDA_API_KEY: https://cloud.lambdalabs.com/api/v1/instance-operations/terminate -d @${INSTANCE_IDS_FILE} -H "Content-Type: application/json" | jq .