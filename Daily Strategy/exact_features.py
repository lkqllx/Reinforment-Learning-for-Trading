# !/usr/bin/env python3
# -*- coding: utf-8 -*-





class DailyBar(object):
    def __init__(self, line_str):
        line_info_list = line_str.strip().split(",")
        self.Date = line_info_list[2]
        self.VWAP = float(line_info_list[3])
        self.todayOpen = float(line_info_list[3])
        self.todayClose = float(line_info_list[6])
        self.zt = self.todayClose - self.todayOpen
        self.zt0 = 0
        self.zt1 = 0
        self.zt2 = 0
        self.zt3 = 0
        self.zt4 = 0
        self.index = 0
    
    def export_line(self):
        export_str = ""
        export_str += str(self.index) + ","
        export_str += str(self.Date) + ","
        export_str += "%.3f," % self.VWAP
        export_str += "%.3f," % self.todayOpen
        export_str += "%.3f," % self.zt
        export_str += "%.3f," % self.zt0
        export_str += "%.3f," % self.zt1
        export_str += "%.3f," % self.zt2
        export_str += "%.3f," % self.zt3
        export_str += "%.3f" % self.zt4
        return export_str


def exact_features(stock_name):
    file_path = "./" + stock_name + "_EODPrices.csv"
    daily_bar_list = []
    index = 0
    with open(file_path) as fr:
        for line in fr.readlines()[1:]:
            daily_bar = DailyBar(line)
            daily_bar.index = index
            if index >= 1:
                daily_bar.zt0 = daily_bar.todayClose - daily_bar_list[-1].todayClose
            if index >= 2:
                daily_bar.zt1 = daily_bar_list[-1].todayClose - daily_bar_list[-2].todayClose
            if index >= 3:
                daily_bar.zt2 = daily_bar_list[-2].todayClose - daily_bar_list[-3].todayClose
            if index >= 4:
                daily_bar.zt3 = daily_bar_list[-3].todayClose - daily_bar_list[-4].todayClose
            if index >= 5:
                daily_bar.zt4 = daily_bar_list[-4].todayClose - daily_bar_list[-5].todayClose
            daily_bar_list.append(daily_bar)
            index += 1
    
    readInCols = ["" ,"Date","VWAP","todayOpen","zt","zt0","zt1","zt2","zt3","zt4"]
    feature_file_path = "./features_" + stock_name + ".csv"
    with open(feature_file_path, 'w+') as fw:
        fw.write(",".join(readInCols) + "\n")
        for daily_bar in daily_bar_list:
            fw.write(daily_bar.export_line() + "\n")
    

if __name__ == "__main__":
    exact_features("510050.SH")