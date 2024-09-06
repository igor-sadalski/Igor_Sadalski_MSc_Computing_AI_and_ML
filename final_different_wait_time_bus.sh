CSV_FILE='final_different_wait_time_bus.csv' 
> "$CSV_FILE"

CHAINS=6

YEAR=2023
MONTH=2
DAY=27
python3 src/Evaluate_routing.py --FINAL_RESULTS_CSV_NAME $CSV_FILE --MAX_WAIT_TIME_INSIDE_BUS 800 --year $YEAR --month $MONTH --day $DAY  & 
python3 src/Evaluate_routing.py --FINAL_RESULTS_CSV_NAME $CSV_FILE --MAX_WAIT_TIME_INSIDE_BUS 800 --year $YEAR --month $MONTH --day $DAY  --IMPERFECT_GENERATIVE_MODEL  &
python3 src/Evaluate_routing.py --FINAL_RESULTS_CSV_NAME $CSV_FILE --MAX_WAIT_TIME_INSIDE_BUS 800 --year $YEAR --month $MONTH --day $DAY  --IMPERFECT_GENERATIVE_MODEL --BETTER_GEN_MODEL &

python3 src/Evaluate_routing.py --FINAL_RESULTS_CSV_NAME $CSV_FILE --MAX_WAIT_TIME_INSIDE_BUS 1000 --year $YEAR --month $MONTH --day $DAY  & 
python3 src/Evaluate_routing.py --FINAL_RESULTS_CSV_NAME $CSV_FILE --MAX_WAIT_TIME_INSIDE_BUS 1000 --year $YEAR --month $MONTH --day $DAY  --IMPERFECT_GENERATIVE_MODEL  &
python3 src/Evaluate_routing.py --FINAL_RESULTS_CSV_NAME $CSV_FILE --MAX_WAIT_TIME_INSIDE_BUS 1000 --year $YEAR --month $MONTH --day $DAY  --IMPERFECT_GENERATIVE_MODEL --BETTER_GEN_MODEL &

python3 src/Evaluate_routing.py --FINAL_RESULTS_CSV_NAME $CSV_FILE --MAX_WAIT_TIME_INSIDE_BUS 900 --year $YEAR --month $MONTH --day $DAY  & 
python3 src/Evaluate_routing.py --FINAL_RESULTS_CSV_NAME $CSV_FILE --MAX_WAIT_TIME_INSIDE_BUS 900 --year $YEAR --month $MONTH --day $DAY  --IMPERFECT_GENERATIVE_MODEL  &
python3 src/Evaluate_routing.py --FINAL_RESULTS_CSV_NAME $CSV_FILE --MAX_WAIT_TIME_INSIDE_BUS 900 --year $YEAR --month $MONTH --day $DAY  --IMPERFECT_GENERATIVE_MODEL --BETTER_GEN_MODEL &

YEAR=2023
MONTH=2
DAY=25
python3 src/Evaluate_routing.py --FINAL_RESULTS_CSV_NAME $CSV_FILE --MAX_WAIT_TIME_INSIDE_BUS 800 --year $YEAR --month $MONTH --day $DAY  & 
python3 src/Evaluate_routing.py --FINAL_RESULTS_CSV_NAME $CSV_FILE --MAX_WAIT_TIME_INSIDE_BUS 800 --year $YEAR --month $MONTH --day $DAY  --IMPERFECT_GENERATIVE_MODEL  &
python3 src/Evaluate_routing.py --FINAL_RESULTS_CSV_NAME $CSV_FILE --MAX_WAIT_TIME_INSIDE_BUS 800 --year $YEAR --month $MONTH --day $DAY  --IMPERFECT_GENERATIVE_MODEL --BETTER_GEN_MODEL &

python3 src/Evaluate_routing.py --FINAL_RESULTS_CSV_NAME $CSV_FILE --MAX_WAIT_TIME_INSIDE_BUS 1000 --year $YEAR --month $MONTH --day $DAY  & 
python3 src/Evaluate_routing.py --FINAL_RESULTS_CSV_NAME $CSV_FILE --MAX_WAIT_TIME_INSIDE_BUS 1000 --year $YEAR --month $MONTH --day $DAY  --IMPERFECT_GENERATIVE_MODEL  &
python3 src/Evaluate_routing.py --FINAL_RESULTS_CSV_NAME $CSV_FILE --MAX_WAIT_TIME_INSIDE_BUS 1000 --year $YEAR --month $MONTH --day $DAY  --IMPERFECT_GENERATIVE_MODEL --BETTER_GEN_MODEL &

python3 src/Evaluate_routing.py --FINAL_RESULTS_CSV_NAME $CSV_FILE --MAX_WAIT_TIME_INSIDE_BUS 900 --year $YEAR --month $MONTH --day $DAY  & 
python3 src/Evaluate_routing.py --FINAL_RESULTS_CSV_NAME $CSV_FILE --MAX_WAIT_TIME_INSIDE_BUS 900 --year $YEAR --month $MONTH --day $DAY  --IMPERFECT_GENERATIVE_MODEL  &
python3 src/Evaluate_routing.py --FINAL_RESULTS_CSV_NAME $CSV_FILE --MAX_WAIT_TIME_INSIDE_BUS 900 --year $YEAR --month $MONTH --day $DAY  --IMPERFECT_GENERATIVE_MODEL --BETTER_GEN_MODEL &

wait 

YEAR=2023
MONTH=2
DAY=26
python3 src/Evaluate_routing.py --FINAL_RESULTS_CSV_NAME $CSV_FILE --MAX_WAIT_TIME_INSIDE_BUS 800 --year $YEAR --month $MONTH --day $DAY  & 
python3 src/Evaluate_routing.py --FINAL_RESULTS_CSV_NAME $CSV_FILE --MAX_WAIT_TIME_INSIDE_BUS 800 --year $YEAR --month $MONTH --day $DAY  --IMPERFECT_GENERATIVE_MODEL  &
python3 src/Evaluate_routing.py --FINAL_RESULTS_CSV_NAME $CSV_FILE --MAX_WAIT_TIME_INSIDE_BUS 800 --year $YEAR --month $MONTH --day $DAY  --IMPERFECT_GENERATIVE_MODEL --BETTER_GEN_MODEL &

python3 src/Evaluate_routing.py --FINAL_RESULTS_CSV_NAME $CSV_FILE --MAX_WAIT_TIME_INSIDE_BUS 1000 --year $YEAR --month $MONTH --day $DAY  & 
python3 src/Evaluate_routing.py --FINAL_RESULTS_CSV_NAME $CSV_FILE --MAX_WAIT_TIME_INSIDE_BUS 1000 --year $YEAR --month $MONTH --day $DAY  --IMPERFECT_GENERATIVE_MODEL  &
python3 src/Evaluate_routing.py --FINAL_RESULTS_CSV_NAME $CSV_FILE --MAX_WAIT_TIME_INSIDE_BUS 1000 --year $YEAR --month $MONTH --day $DAY  --IMPERFECT_GENERATIVE_MODEL --BETTER_GEN_MODEL &

python3 src/Evaluate_routing.py --FINAL_RESULTS_CSV_NAME $CSV_FILE --MAX_WAIT_TIME_INSIDE_BUS 900 --year $YEAR --month $MONTH --day $DAY  & 
python3 src/Evaluate_routing.py --FINAL_RESULTS_CSV_NAME $CSV_FILE --MAX_WAIT_TIME_INSIDE_BUS 900 --year $YEAR --month $MONTH --day $DAY  --IMPERFECT_GENERATIVE_MODEL  &
python3 src/Evaluate_routing.py --FINAL_RESULTS_CSV_NAME $CSV_FILE --MAX_WAIT_TIME_INSIDE_BUS 900 --year $YEAR --month $MONTH --day $DAY  --IMPERFECT_GENERATIVE_MODEL --BETTER_GEN_MODEL &


YEAR=2023
MONTH=2
DAY=28
python3 src/Evaluate_routing.py --FINAL_RESULTS_CSV_NAME $CSV_FILE --MAX_WAIT_TIME_INSIDE_BUS 800 --year $YEAR --month $MONTH --day $DAY  & 
python3 src/Evaluate_routing.py --FINAL_RESULTS_CSV_NAME $CSV_FILE --MAX_WAIT_TIME_INSIDE_BUS 800 --year $YEAR --month $MONTH --day $DAY  --IMPERFECT_GENERATIVE_MODEL  &
python3 src/Evaluate_routing.py --FINAL_RESULTS_CSV_NAME $CSV_FILE --MAX_WAIT_TIME_INSIDE_BUS 800 --year $YEAR --month $MONTH --day $DAY  --IMPERFECT_GENERATIVE_MODEL --BETTER_GEN_MODEL &

python3 src/Evaluate_routing.py --FINAL_RESULTS_CSV_NAME $CSV_FILE --MAX_WAIT_TIME_INSIDE_BUS 1000 --year $YEAR --month $MONTH --day $DAY  & 
python3 src/Evaluate_routing.py --FINAL_RESULTS_CSV_NAME $CSV_FILE --MAX_WAIT_TIME_INSIDE_BUS 1000 --year $YEAR --month $MONTH --day $DAY  --IMPERFECT_GENERATIVE_MODEL  &
python3 src/Evaluate_routing.py --FINAL_RESULTS_CSV_NAME $CSV_FILE --MAX_WAIT_TIME_INSIDE_BUS 1000 --year $YEAR --month $MONTH --day $DAY  --IMPERFECT_GENERATIVE_MODEL --BETTER_GEN_MODEL &

python3 src/Evaluate_routing.py --FINAL_RESULTS_CSV_NAME $CSV_FILE --MAX_WAIT_TIME_INSIDE_BUS 900 --year $YEAR --month $MONTH --day $DAY  & 
python3 src/Evaluate_routing.py --FINAL_RESULTS_CSV_NAME $CSV_FILE --MAX_WAIT_TIME_INSIDE_BUS 900 --year $YEAR --month $MONTH --day $DAY  --IMPERFECT_GENERATIVE_MODEL  &
python3 src/Evaluate_routing.py --FINAL_RESULTS_CSV_NAME $CSV_FILE --MAX_WAIT_TIME_INSIDE_BUS 900 --year $YEAR --month $MONTH --day $DAY  --IMPERFECT_GENERATIVE_MODEL --BETTER_GEN_MODEL &

wait

echo "===============================All scripts for all months have finished running."



