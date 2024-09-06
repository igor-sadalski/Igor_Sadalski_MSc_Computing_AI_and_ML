CSV_FILE='final_results.csv' 
> "$CSV_FILE"

CHAINS=6

YEAR=2023
MONTH=2
DAY=27
python3 src/Evaluate_routing.py --FINAL_RESULTS_CSV_NAME $CSV_FILE --mode 1 --year $YEAR --month $MONTH --day $DAY &
python3 src/Evaluate_routing.py --FINAL_RESULTS_CSV_NAME $CSV_FILE --mode 2 --year $YEAR --month $MONTH --day $DAY --N_CHAINS $CHAINS & 
python3 src/Evaluate_routing.py --FINAL_RESULTS_CSV_NAME $CSV_FILE --mode 2 --year $YEAR --month $MONTH --day $DAY --N_CHAINS $CHAINS --IMPERFECT_GENERATIVE_MODEL &
python3 src/Evaluate_routing.py --FINAL_RESULTS_CSV_NAME $CSV_FILE --mode 2 --year $YEAR --month $MONTH --day $DAY --N_CHAINS $CHAINS --IMPERFECT_GENERATIVE_MODEL --BETTER_GEN_MODEL &

YEAR=2023
MONTH=2
DAY=25
python3 src/Evaluate_routing.py --FINAL_RESULTS_CSV_NAME $CSV_FILE --mode 1 --year $YEAR --month $MONTH --day $DAY &
python3 src/Evaluate_routing.py --FINAL_RESULTS_CSV_NAME $CSV_FILE --mode 2 --year $YEAR --month $MONTH --day $DAY --N_CHAINS $CHAINS & 
python3 src/Evaluate_routing.py --FINAL_RESULTS_CSV_NAME $CSV_FILE --mode 2 --year $YEAR --month $MONTH --day $DAY --N_CHAINS $CHAINS --IMPERFECT_GENERATIVE_MODEL &
python3 src/Evaluate_routing.py --FINAL_RESULTS_CSV_NAME $CSV_FILE --mode 2 --year $YEAR --month $MONTH --day $DAY --N_CHAINS $CHAINS --IMPERFECT_GENERATIVE_MODEL --BETTER_GEN_MODEL &

YEAR=2023
MONTH=2
DAY=26
python3 src/Evaluate_routing.py --FINAL_RESULTS_CSV_NAME $CSV_FILE --mode 1 --year $YEAR --month $MONTH --day $DAY &
python3 src/Evaluate_routing.py --FINAL_RESULTS_CSV_NAME $CSV_FILE --mode 2 --year $YEAR --month $MONTH --day $DAY --N_CHAINS $CHAINS & 
python3 src/Evaluate_routing.py --FINAL_RESULTS_CSV_NAME $CSV_FILE --mode 2 --year $YEAR --month $MONTH --day $DAY --N_CHAINS $CHAINS --IMPERFECT_GENERATIVE_MODEL &
python3 src/Evaluate_routing.py --FINAL_RESULTS_CSV_NAME $CSV_FILE --mode 2 --year $YEAR --month $MONTH --day $DAY --N_CHAINS $CHAINS --IMPERFECT_GENERATIVE_MODEL --BETTER_GEN_MODEL &

YEAR=2023
MONTH=2
DAY=28
python3 src/Evaluate_routing.py --FINAL_RESULTS_CSV_NAME $CSV_FILE --mode 1 --year $YEAR --month $MONTH --day $DAY &
python3 src/Evaluate_routing.py --FINAL_RESULTS_CSV_NAME $CSV_FILE --mode 2 --year $YEAR --month $MONTH --day $DAY --N_CHAINS $CHAINS & 
python3 src/Evaluate_routing.py --FINAL_RESULTS_CSV_NAME $CSV_FILE --mode 2 --year $YEAR --month $MONTH --day $DAY --N_CHAINS $CHAINS --IMPERFECT_GENERATIVE_MODEL &
python3 src/Evaluate_routing.py --FINAL_RESULTS_CSV_NAME $CSV_FILE --mode 2 --year $YEAR --month $MONTH --day $DAY --N_CHAINS $CHAINS --IMPERFECT_GENERATIVE_MODEL --BETTER_GEN_MODEL &

wait

echo "===============================All scripts for all months have finished running."



