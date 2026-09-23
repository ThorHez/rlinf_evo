# Source this file: source /path/to/rlinf_evo/local_setup/activate.sh
_rlinf_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "${_rlinf_repo}/.venv/bin/activate"
export OPENPI_DATA_HOME="${_rlinf_repo}/.venv/assets/openpi"
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
# Keep Python 3.10 ROS paths from the login shell out of this Python 3.11 env.
export PYTHONNOUSERSITE=1
export PYTHONPATH="${_rlinf_repo}"
unset _rlinf_repo
