import optparse
import os
import datetime
import random
import traceback
import osmnx as ox
import numpy as np
import pandas as pd

from Data_structures import Data_folders, Date_operational_range, Completed_requests_stats, Failed_requests_stats
# from benchmark_1.utilities import log_runtime_and_memory

class Request_handler:

    def __init__(self, data_folders: Data_folders, process_requests: bool,
                 default_start_hour: int = 19, default_end_hour: int = 2):
        self.request_folder_path = data_folders.request_folder_path
        self.routing_data_folder = data_folders.routing_data_folder
        self.processed_requests_folder_path = data_folders.processed_requests_folder_path

        self.columns_of_interest = ["Request Creation Date", "Request Creation Time", "Number of Passengers", "Booking Type", 
                                    "Requested Pickup Time", "Origin Lat", "Origin Lng", "Destination Lat", "Destination Lng", 
                                    "Request Status", "On-demand ETA", "Ride Duration"]
        self.status_of_interest = ["Completed", "Unaccepted Proposal", "Cancel"]
        self.G = self._initialize_map()
        self.month_lengths = {2022: [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
                              2023: [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]}
        self.default_start_hour = default_start_hour
        self.default_end_hour = default_end_hour
        self.default_start_hour = default_start_hour
        self.default_end_hour = default_end_hour

        if process_requests:
            self.scheduled_requests_df, self.online_requests_df = self._initialize_dataframes()
            self._filter_requests_dataframes()
            self._save_requests_dataframes()
        else:
            self.scheduled_requests_df, self.online_requests_df = self._load_requests_dataframes()
            self._correct_datetime_for_dataframes()
        

    def _initialize_map(self):
        graphml_filepath = os.path.join(self.routing_data_folder, "graph_structure.graphml")
        G = ox.io.load_graphml(graphml_filepath)
        return G
    
    def _load_requests_dataframes(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        scheduled_requests_path = os.path.join(self.processed_requests_folder_path, "scheduled_requests.csv")
        scheduled_requests_df = pd.read_csv(scheduled_requests_path)

        online_requests_path = os.path.join(self.processed_requests_folder_path, "online_requests.csv")
        online_requests_df = pd.read_csv(online_requests_path)
        
        return scheduled_requests_df, online_requests_df
    
    def _correct_datetime_for_dataframes(self):
        self.scheduled_requests_df["Request Creation Time"] = pd.to_datetime(self.scheduled_requests_df["Request Creation Time"])
        self.scheduled_requests_df["Requested Pickup Time"] = pd.to_datetime(self.scheduled_requests_df["Requested Pickup Time"])
        self.scheduled_requests_df["Request Creation Date"] = pd.to_datetime(self.scheduled_requests_df["Request Creation Date"])
        self.online_requests_df["Request Creation Time"] = pd.to_datetime(self.online_requests_df["Request Creation Time"])
        self.online_requests_df["Requested Pickup Time"] = pd.to_datetime(self.online_requests_df["Requested Pickup Time"])
        self.online_requests_df["Request Creation Date"] = pd.to_datetime(self.online_requests_df["Request Creation Date"])
    
    def _remove_dates_with_not_enough_requests(self):
        dates_to_test = self.online_requests_df["Request Creation Date"].unique()

        for test_date in dates_to_test:
            date_operational_range = Date_operational_range(year = test_date.year,
                                                            month = test_date.month,
                                                            day = test_date.day,
                                                            start_hour = self.default_start_hour,
                                                            end_hour = self.default_end_hour)
            _, online_requests = self.get_requests_for_given_date_and_hour_range(date_operational_range=date_operational_range)

            if len(online_requests.index) < 10:
                online_drop_condition = (self.online_requests_df["Request Creation Date"] == test_date)
                scheduled_drop_condition = (self.scheduled_requests_df["Request Creation Date"] == test_date)
                self.online_requests_df = self.online_requests_df[~online_drop_condition]
                self.scheduled_requests_df = self.scheduled_requests_df[~scheduled_drop_condition]
            
    
    def _filter_requests_dataframes(self):
        sched_requests_mask = self.scheduled_requests_df["Request Status"].isin(self.status_of_interest)
        online_requests_maks = self.online_requests_df["Request Status"].isin(self.status_of_interest)

        self.scheduled_requests_df = self.scheduled_requests_df[sched_requests_mask]
        self.online_requests_df = self.online_requests_df[online_requests_maks]
        
        self.scheduled_requests_df["Number of Passengers"] = 1
        self.online_requests_df["Number of Passengers"] = 1
        self._remove_dates_with_not_enough_requests()

    def _save_requests_dataframes(self):
        scheduled_requests_path = os.path.join(self.processed_requests_folder_path, "scheduled_requests.csv")
        self.scheduled_requests_df.to_csv(scheduled_requests_path, index=True)
        
        online_requests_path = os.path.join(self.processed_requests_folder_path, "online_requests.csv")
        self.online_requests_df.to_csv(online_requests_path, index=True)
    
    def _initialize_dataframes(self):
        combined_requests_df = self._read_xlsx_files_into_data_frame()
        scheduled_requests_df, online_requests_df = self._divide_requests_dataframe(dataframe=combined_requests_df)
        processed_scheduled_requests_df = self._process_requests_dataframe(dataframe=scheduled_requests_df)
        processed_online_requests_df = self._process_requests_dataframe(dataframe=online_requests_df)

        return processed_scheduled_requests_df, processed_online_requests_df

    def _read_xlsx_files_into_data_frame(self):
        dataframes_list = []
        for filename in os.listdir(self.request_folder_path):
            requests_filepath = os.path.join(self.request_folder_path, filename)
            df = pd.read_excel(requests_filepath)
            dataframes_list.append(df)
        
        combined_requests_df = pd.concat(dataframes_list)
        combined_requests_df.reset_index(drop=True, inplace=True)

        return combined_requests_df
    
    def _divide_requests_dataframe(self, dataframe):
        filtered_requests_df = dataframe[self.columns_of_interest]
        scheduled_requests_df = filtered_requests_df[(filtered_requests_df["Booking Type"] == "Prebooking")]
        online_requests_df = filtered_requests_df[(filtered_requests_df["Booking Type"] == "On Demand")]

        return scheduled_requests_df, online_requests_df 
    
    def _get_nodes_from_coordinates(self, dataframe):
        origin_nodes = ox.nearest_nodes(self.G, dataframe["Origin Lng"], dataframe["Origin Lat"])
        destination_nodes = ox.nearest_nodes(self.G, dataframe["Destination Lng"], dataframe["Destination Lat"])

        return origin_nodes, destination_nodes
    
    def _process_requests_dataframe(self, dataframe):
        origin_nodes, destination_nodes = self._get_nodes_from_coordinates(dataframe=dataframe)
        processed_requests_df = dataframe.copy()
        processed_requests_df["Origin Node"] = origin_nodes
        processed_requests_df["Destination Node"] = destination_nodes

        return processed_requests_df
    
    
    def get_requests_for_given_date_and_hour_range(self, date_operational_range: Date_operational_range):
        if date_operational_range.end_hour < date_operational_range.start_hour:
            current_date_string = str(date_operational_range.year)+"-"+str(date_operational_range.month)+"-"+str(date_operational_range.day)
            current_date_object = pd.to_datetime(current_date_string).date()

            current_scheduled_mask = (self.scheduled_requests_df["Requested Pickup Time"].dt.date == current_date_object) \
                & (self.scheduled_requests_df["Requested Pickup Time"].dt.hour >= date_operational_range.start_hour) \
                & (self.scheduled_requests_df["Requested Pickup Time"].dt.hour <= 23)
            
            current_online_mask = (self.online_requests_df["Requested Pickup Time"].dt.date == current_date_object) \
                & (self.online_requests_df["Requested Pickup Time"].dt.hour >= date_operational_range.start_hour) \
                & (self.online_requests_df["Requested Pickup Time"].dt.hour <= 23)
            
            if date_operational_range.day+1 > self.month_lengths[date_operational_range.year][date_operational_range.month-1]:
                following_day = 1
                if date_operational_range.month + 1 > 12:
                    following_month = 1
                    following_year = date_operational_range.year + 1
                else:
                    following_month = date_operational_range.month + 1
                    following_year = date_operational_range.year
            else:
                following_day = date_operational_range.day+1
                following_month = date_operational_range.month
                following_year = date_operational_range.year
            following_date_string = str(following_year)+"-"+str(following_month)+"-"+str(following_day)
            following_date_object = pd.to_datetime(following_date_string).date()

            following_scheduled_mask = (self.scheduled_requests_df["Requested Pickup Time"].dt.date == following_date_object) \
                & (self.scheduled_requests_df["Requested Pickup Time"].dt.hour >= 0) \
                & (self.scheduled_requests_df["Requested Pickup Time"].dt.hour <= date_operational_range.end_hour)
            
            following_online_mask = (self.online_requests_df["Requested Pickup Time"].dt.date == following_date_object) \
                & (self.online_requests_df["Requested Pickup Time"].dt.hour >= 0) \
                & (self.online_requests_df["Requested Pickup Time"].dt.hour <= date_operational_range.end_hour)
            
            scheduled_requests_df = self.scheduled_requests_df[current_scheduled_mask | following_scheduled_mask]
            scheduled_requests_df = scheduled_requests_df.sort_values(by=["Requested Pickup Time"]).sort_index()
            online_requests_df = self.online_requests_df[current_online_mask | following_online_mask]
            online_requests_df = online_requests_df.sort_values(by=["Requested Pickup Time"]).sort_index()
        else:
            date_string = str(date_operational_range.year)+"-"+str(date_operational_range.month)+"-"+str(date_operational_range.day)
            date_object = pd.to_datetime(date_string).date()

            scheduled_mask = (self.scheduled_requests_df["Requested Pickup Time"].dt.date == date_object) \
                & (self.scheduled_requests_df["Requested Pickup Time"].dt.hour >= date_operational_range.start_hour) \
                & (self.scheduled_requests_df["Requested Pickup Time"].dt.hour <= date_operational_range.end_hour)
            
            online_mask = (self.online_requests_df["Requested Pickup Time"].dt.date == date_object) \
                & (self.online_requests_df["Requested Pickup Time"].dt.hour >= date_operational_range.start_hour) \
                & (self.online_requests_df["Requested Pickup Time"].dt.hour <= date_operational_range.end_hour)
            
            scheduled_requests_df = self.scheduled_requests_df[scheduled_mask]
            scheduled_requests_df = scheduled_requests_df.sort_values(by=["Requested Pickup Time"]).sort_index()
            online_requests_df = self.online_requests_df[online_mask]
            online_requests_df = online_requests_df.sort_values(by=["Requested Pickup Time"]).sort_index()
        
        return scheduled_requests_df, online_requests_df
    
    def get_requests_for_given_minute_range(self, year: int, month: int, day: int, hour: int, 
                                            start_minute: int, end_minute: int):
        date_string = str(year)+"-"+str(month)+"-"+str(day)
        date_object = pd.to_datetime(date_string).date()

        scheduled_mask = (self.scheduled_requests_df["Requested Pickup Time"].dt.date == date_object) \
            & (self.scheduled_requests_df["Requested Pickup Time"].dt.hour == hour) \
            & (self.scheduled_requests_df["Requested Pickup Time"].dt.minute >= start_minute) \
            & (self.scheduled_requests_df["Requested Pickup Time"].dt.minute <= end_minute)
        
        online_mask = (self.online_requests_df["Requested Pickup Time"].dt.date == date_object) \
            & (self.online_requests_df["Requested Pickup Time"].dt.hour == hour) \
            & (self.online_requests_df["Requested Pickup Time"].dt.minute >= start_minute) \
            & (self.online_requests_df["Requested Pickup Time"].dt.minute <= end_minute)
        
        return self.scheduled_requests_df[scheduled_mask], self.online_requests_df[online_mask]
    
    # @log_runtime_and_memory
    def get_requests_before_given_date(self, year: int, month: int, day: int) -> pd.DataFrame:
        '''get all historical values that happened before start of our system 
        return them as a merged values'''
        date_string = str(year)+"-"+str(month)+"-"+str(day)
        date_object = pd.to_datetime(date_string).date()

        online_mask = (self.online_requests_df["Requested Pickup Time"].dt.date < date_object)
        
        online_requests = self.online_requests_df[online_mask]
        
        return online_requests
    
    def get_online_requests_for_given_minute(self, date_operational_range: Date_operational_range, year: int, 
                                             month: int, day: int, hour: int, minute: int):
        date_string = str(year)+"-"+str(month)+"-"+str(day)
        date_object = pd.to_datetime(date_string).date()

        if date_operational_range.end_hour < date_operational_range.start_hour:
            if hour < date_operational_range.end_hour:
                hour_boundary = date_operational_range.end_hour
            else:
                hour_boundary = 23
        else:
            hour_boundary = date_operational_range.end_hour
        
        online_mask = (self.online_requests_df["Request Creation Time"].dt.date == date_object) \
            & (self.online_requests_df["Request Creation Time"].dt.hour == hour) \
            & (self.online_requests_df["Request Creation Time"].dt.minute == minute) \
            & (self.online_requests_df["Requested Pickup Time"].dt.date == date_object) \
            & (self.online_requests_df["Requested Pickup Time"].dt.hour <= hour_boundary)
        
        online_requests_df = self.online_requests_df[online_mask]
        
        return online_requests_df
    
    # def get_next_n_requests(date_operational_range, year,month, day,hour, minute):

    
    def get_initial_requests(self, date_operational_range: Date_operational_range):
        if date_operational_range.end_hour < date_operational_range.start_hour:
            current_date_string = str(date_operational_range.year)+"-"+str(date_operational_range.month)+"-"+str(date_operational_range.day)
            current_date_object = pd.to_datetime(current_date_string).date()

            if date_operational_range.day+1 > self.month_lengths[date_operational_range.year][date_operational_range.month-1]:
                following_day = 1
                if date_operational_range.month + 1 > 12:
                    following_month = 1
                    following_year = date_operational_range.year + 1
                else:
                    following_month = date_operational_range.month + 1
                    following_year = date_operational_range.year
            else:
                following_day = date_operational_range.day+1
                following_month = date_operational_range.month
                following_year = date_operational_range.year
            following_date_string = str(following_year)+"-"+str(following_month)+"-"+str(following_day)
            following_date_object = pd.to_datetime(following_date_string).date()

            current_scheduled_mask = (self.scheduled_requests_df["Requested Pickup Time"].dt.date == current_date_object) \
                            & (self.scheduled_requests_df["Requested Pickup Time"].dt.hour >= date_operational_range.start_hour) \
                            & (self.scheduled_requests_df["Requested Pickup Time"].dt.hour <= 23)
            
            current_online_mask = (self.online_requests_df["Request Creation Time"].dt.date == current_date_object) \
                & (self.online_requests_df["Request Creation Time"].dt.hour < date_operational_range.start_hour) \
                & (self.online_requests_df["Requested Pickup Time"].dt.date == current_date_object) \
                & (self.online_requests_df["Requested Pickup Time"].dt.hour >= date_operational_range.start_hour) \
                & (self.online_requests_df["Requested Pickup Time"].dt.hour <= 23)
            
            following_scheduled_mask = (self.scheduled_requests_df["Requested Pickup Time"].dt.date == following_date_object) \
                            & (self.scheduled_requests_df["Requested Pickup Time"].dt.hour >= 0) \
                            & (self.scheduled_requests_df["Requested Pickup Time"].dt.hour <= date_operational_range.end_hour)
            
            following_online_mask = (self.online_requests_df["Request Creation Time"].dt.date == current_date_object) \
                & (self.online_requests_df["Request Creation Time"].dt.hour < date_operational_range.start_hour) \
                & (self.online_requests_df["Requested Pickup Time"].dt.date == following_date_object) \
                & (self.online_requests_df["Requested Pickup Time"].dt.hour >= 0) \
                & (self.online_requests_df["Requested Pickup Time"].dt.hour <= date_operational_range.end_hour)
            
            online_requests_df = self.online_requests_df[current_online_mask | following_online_mask]
            scheduled_requests_df = self.scheduled_requests_df[current_scheduled_mask | following_scheduled_mask]
            
            combined_requests_df = pd.concat([scheduled_requests_df, online_requests_df])
        
        else:
            date_string = str(date_operational_range.year)+"-"+str(date_operational_range.month)+"-"+str(date_operational_range.day)
            date_object = pd.to_datetime(date_string).date()

            scheduled_mask = (self.scheduled_requests_df["Requested Pickup Time"].dt.date == date_object) \
                            & (self.scheduled_requests_df["Requested Pickup Time"].dt.hour >= date_operational_range.start_hour) \
                            & (self.scheduled_requests_df["Requested Pickup Time"].dt.hour <= date_operational_range.end_hour)
            
            online_mask = (self.online_requests_df["Request Creation Time"].dt.date == date_object) \
                & (self.online_requests_df["Request Creation Time"].dt.hour < date_operational_range.start_hour) \
                & (self.online_requests_df["Requested Pickup Time"].dt.date == date_object) \
                & (self.online_requests_df["Requested Pickup Time"].dt.hour >= date_operational_range.start_hour) \
                & (self.online_requests_df["Requested Pickup Time"].dt.hour <= date_operational_range.end_hour)
            
            online_requests_df = self.online_requests_df[online_mask]
            scheduled_requests_df = self.scheduled_requests_df[scheduled_mask]
            
            combined_requests_df = pd.concat([scheduled_requests_df, online_requests_df])
        
        return combined_requests_df
    
    def generate_operating_ranges(self, date_operational_range: Date_operational_range):
        if date_operational_range.end_hour < date_operational_range.start_hour:
            current_hour_range = list(range(date_operational_range.start_hour, 24))
            first_day_range = [date_operational_range.day] * len(current_hour_range)
            first_month_range = [date_operational_range.month] * len(current_hour_range)
            first_year_range = [date_operational_range.year] * len(current_hour_range)

            following_hour_range = list(range(0, date_operational_range.end_hour+1))
            hour_range = current_hour_range + following_hour_range

            if date_operational_range.day+1 > self.month_lengths[date_operational_range.year][date_operational_range.month-1]:
                if date_operational_range.month + 1 > 12:
                    second_day_range = [1] * len(following_hour_range)
                    second_month_range = [1] * len(following_hour_range)
                    second_year_range = [date_operational_range.year + 1] * len(following_hour_range)
                else:
                    second_day_range = [1] * len(following_hour_range)
                    second_month_range = [date_operational_range.month + 1] * len(following_hour_range)
                    second_year_range = [date_operational_range.year] * len(following_hour_range)
            else:
                second_day_range = [date_operational_range.day + 1] * len(following_hour_range)
                second_month_range = [date_operational_range.month] * len(following_hour_range)
                second_year_range = [date_operational_range.year] * len(following_hour_range)
            day_range = first_day_range + second_day_range
            month_range = first_month_range + second_month_range
            year_range = first_year_range + second_year_range
        else:
            hour_range = list(range(date_operational_range.start_hour, date_operational_range.end_hour+1))
            day_range = [date_operational_range.day] * len(hour_range)
            month_range = [date_operational_range.month] * len(hour_range)
            year_range = [date_operational_range.year] * len(hour_range)

        
        return hour_range, day_range, month_range, year_range
    
    def extract_data_statistics_for_given_date(self, date_operational_range: Date_operational_range):
        scheduled_requests_df, online_requests_df = self.get_requests_for_given_date_and_hour_range(date_operational_range=date_operational_range)
        combined_requests_df = pd.concat([scheduled_requests_df, online_requests_df])

        print("Request info: " + str(len(combined_requests_df.index)))

        completed_requests_stats = Completed_requests_stats(combined_requests_df=combined_requests_df)
        failed_requests_stats = Failed_requests_stats(combined_requests_df=combined_requests_df)

        return completed_requests_stats, failed_requests_stats
    
    def generate_minute_intervals(self, interval_length: int = 5):
        num_intervals = 60 // interval_length

        minute_intervals = []
        for i in range(num_intervals):
            minute_intervals.append(((i*interval_length), ((i+1)*(interval_length))-1))

        return minute_intervals
    
    def generate_testing_dates(self, save_testing_dates: bool = False):
        random.seed(42)
        unique_date_elements = list(self.online_requests_df["Request Creation Date"].unique())
        date_series = pd.to_datetime(pd.Series(unique_date_elements))
        grouped = date_series.groupby([date_series.dt.year, date_series.dt.month])
        def select_two_random(group):
            if len(group) < 2:
                return group
            return group.sample(n=2, random_state=42)
        selected_dates = grouped.apply(select_two_random)
        selected_dates = selected_dates.reset_index(drop=True)

        if save_testing_dates:
            selected_dates.to_csv("data/test_days.csv", index=False)

        # Convert to list and return
        return selected_dates.tolist()
    
    def extract_metadata(self, start_hour: int = 19 , end_hour: int = 2, interval_length: int = 5):
        combined_requests_df = pd.concat([self.scheduled_requests_df, self.online_requests_df])

        unique_count = combined_requests_df["Request Creation Date"].nunique()

        unique_date_elements = combined_requests_df["Request Creation Date"].unique()

        minute_intervals = self.generate_minute_intervals(interval_length=interval_length)

        max_num_of_requests_in_interval = 0
        total_num_requests_in_interval = 0
        num_requests_in_interval_dict = {}
        number_of_intervals = 0

        for date_element in unique_date_elements:
            date_operational_range = Date_operational_range(year=date_element.year,
                                                            month=date_element.month,
                                                            day=date_element.day,
                                                            start_hour=start_hour,
                                                            end_hour=end_hour)
            hour_range, day_range, month_range, year_range = self.generate_operating_ranges(date_operational_range=date_operational_range)

            for i, hour_of_interest in enumerate(hour_range):
                new_date_string = str(year_range[i])+"-"+str(month_range[i])+"-"+str(day_range[i])
                new_date_object = pd.to_datetime(new_date_string).date()

                for minute_interval in minute_intervals:
                    retrieval_mask = (combined_requests_df["Requested Pickup Time"].dt.date == new_date_object) \
                        & (combined_requests_df["Requested Pickup Time"].dt.hour == hour_of_interest) \
                        & (combined_requests_df["Requested Pickup Time"].dt.minute >= minute_interval[0]) \
                        & (combined_requests_df["Requested Pickup Time"].dt.minute <= minute_interval[1])
                    
                    retrieved_requests = combined_requests_df[retrieval_mask]
                    number_of_requests_in_interval = len(retrieved_requests.index)

                    number_of_intervals += 1
                    total_num_requests_in_interval += number_of_requests_in_interval

                    if number_of_requests_in_interval > max_num_of_requests_in_interval:
                        max_num_of_requests_in_interval = number_of_requests_in_interval
                    
                    if number_of_requests_in_interval in num_requests_in_interval_dict:
                        num_requests_in_interval_dict[number_of_requests_in_interval] += 1
                    else:
                        num_requests_in_interval_dict[number_of_requests_in_interval] = 1
            
            avg_num_requests_in_interval = total_num_requests_in_interval / number_of_intervals

        return unique_count, max_num_of_requests_in_interval, avg_num_requests_in_interval, num_requests_in_interval_dict
        
 
if __name__ == '__main__':
    """Performs execution delta of the process."""
    # Unit tests
    pStart = datetime.datetime.now()
    try:
        parser = optparse.OptionParser()
        parser.add_option("--year", type=int, dest='year', default=2022, help='Select year of interest.')
        parser.add_option("--month", type=int, dest='month', default=8, help='Select month of interest.')
        parser.add_option("--day", type=int, dest='day', default=17, help='Select day of interest.')
        parser.add_option("--start_hour", type=int, dest='start_hour', default=19, help='Select start hour for the time range of interest') #19
        parser.add_option("--end_hour", type=int, dest='end_hour', default=2, help='Select end hour for the time range of interest')
        parser.add_option("--interval_length", type=int, dest='interval_length', default=5, help='Select interval length for retrieving metadata')
        (options, args) = parser.parse_args()
        
        data_folders = Data_folders()
        date_operational_range = Date_operational_range(year=options.year, 
                                                    month=options.month,
                                                    day=options.day,
                                                    start_hour=options.start_hour,
                                                    end_hour=options.end_hour)

        rqh = Request_handler(data_folders=data_folders, process_requests=True)
        rqh.generate_testing_dates(save_testing_dates=True)
        # metadata_results = rqh.extract_metadata(start_hour=options.start_hour, 
        #                                        end_hour=options.end_hour,
        #                                        interval_length=options.interval_length)
        # number_of_days_in_the_data, max_requests_in_interval, avg_num_requests_in_interval, num_requests_dict = metadata_results
        # print("Number of days in the data loaded = " + str(number_of_days_in_the_data))
        # print("Maximum number of requests in " + str(options.interval_length) + "-minute intervals = " + str(max_requests_in_interval))
        # print("Average number of requests in " + str(options.interval_length) + "-minute interval = " + str(avg_num_requests_in_interval))
        # print("Frequency counts for different numbers of requests in " + str(options.interval_length) + "-minute intervals = " + str(num_requests_dict))
        # completed_requests_stats, failed_requests_stats = rqh.extract_data_statistics_for_given_date(date_operational_range=date_operational_range)

        # print(completed_requests_stats)
        # print(failed_requests_stats)

        # failed_requests_avg_wait_time = failed_requests_stats.avg_wait_time_for_failed_requests

        # count_of_potentially_failed_requests = 0
        # for wait_time_value in failed_requests_stats.wait_times_for_canceled_requests:
        #     if wait_time_value > failed_requests_avg_wait_time:
        #         count_of_potentially_failed_requests += 1
        
        # for wait_time_value in completed_requests_stats.passenger_wait_times:
        #     if wait_time_value > failed_requests_avg_wait_time:
        #         count_of_potentially_failed_requests += 1

        # print("Number of requests with higher wait times than the average wait time of real cancelled requests = " + str(count_of_potentially_failed_requests))
        

    except Exception as errorMainContext:
        print("Fail End Process: ", errorMainContext)
        traceback.print_exc()
    qStop = datetime.datetime.now()
    print("Execution time: " + str(qStop-pStart))

        
