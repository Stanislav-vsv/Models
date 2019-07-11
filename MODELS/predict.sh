#!/bin/bash

# ---------------------------------------------------------------------------------
# This is typical block for environment variables setup. Please do not edit it. 

cd /home/informatica/jupyter-notebook/GuaranteesModel

if [ -z "$MODELS_ENVIRONMENT_PATH" ];
then MODELS_ENVIRONMENT_PATH=../.. ;
fi;


echo "MODELS_ENVIRONMENT_PATH is set to '$MODELS_ENVIRONMENT_PATH'";
source $MODELS_ENVIRONMENT_PATH/models.environment.sh;

login_var="${PWD##*/}_LOGIN";
password_var="${PWD##*/}_PASSWORD";

if [ -z "$1"  ];
then
    TIME=$(date +%y%m%d%H%M%S)"00";
    OUT=`cksum << EOF
    ${PWD##*/}
    EOF`;
    input=$OUT;
    len=${#input};
    out_rev="";
    for ((i=$len-4; i>=$len-7; i--))
    do
	out_rev="$out_rev${input:$i:1}"
    done
    JOB=$TIME$out_rev;
else
    JOB=$1;
fi;

# Now you can use variables ${!login_var}, ${!password_var} and $CONNECT_STRING for connection to Database
# And variable $JOB for logging run processes
#----------------------------------------------------------------------------------

#sqlplus ${!login_var}/${!password_var}@$CONNECT_STRING @Guarantees_dataset.sql
# "'$PARAM'"

export PATH=/informatica/anaconda2/bin:$PATH
python  model_guarantees.py --login=${!login_var} --password=${!password_var} --tns=$CONNECT_STRING --job=$JOB
