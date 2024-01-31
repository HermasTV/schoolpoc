cleanup() {
    echo "Terminating background processes..."
    # Send interrupt signal to the specific process group (PGID) of the background processes
    kill -- -$$
    exit 0
}
trap cleanup INT

echo "Press Ctrl + C to terminate"
python school.py &
streamlit run streamlit.py &
wait