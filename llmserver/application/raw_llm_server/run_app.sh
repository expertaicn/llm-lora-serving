basepath=$(cd `dirname $0`; pwd)
echo "basepath: $basepath"

export PYTHONPATH=${basepath}/../../ && streamlit run app.py   --server.port 8541
#streamlit run baichun_13_demo.py