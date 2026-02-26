if python -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
    echo "Python version is sufficient (>= 3.10)."
else
    echo "Error: Python version is less than 3.10."
    exit 1
fi

pip install virtualenv

virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
deactivate