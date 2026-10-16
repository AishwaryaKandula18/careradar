"""One command: python run_all.py [data_dir]  -> trains model, builds dashboard."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import train, build_dashboard
d = sys.argv[1] if len(sys.argv) > 1 else "data"
train.main(d); build_dashboard.main(d)
