import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import geodatasets
import seaborn as sns


def parseVIZ(filename):
    column = False
    columns = []
    data = []
    with open(filename, encoding="UTF-8") as f:
        for line in f:
            if not line.startswith("//"):
                if column:
                    data.append(line[:-1].split("\t"))
                else:
                    columns = (line[:-1].split("\t"))
                    column = True
    return pd.DataFrame(data, columns=columns)


def parseARGO(filename):
    column = False
    columns = []
    data = []
    with open(filename, encoding="UTF-8") as f:
        for line in f:
            if not line.startswith("#"):
                if column:
                    data.append(line[:-1].split(","))
                else:
                    columns = (line[:-1].split(","))
                    column = True

    output = pd.DataFrame(data, columns=columns)
    output["parameters"] = output["parameters"].apply(lambda x: x.split(" "))
    output["date"] = pd.to_datetime(output["date"], format="%Y%m%d%H%M%S")
    return output


def argoPList(df: pd.DataFrame):
    pars = set()
    df["parameters"].apply(lambda x: pars.update(x))
    return list(pars)


#Nitrate[µmol/kg]
def answerQ1(df: pd.DataFrame):
    fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True)

    df["Nitrate[µmol/kg]"] = pd.to_numeric(df["Nitrate[µmol/kg]"])
    df["Station"] = pd.to_numeric(df["Station"])
    df[df["Nitrate[µmol/kg]"] > -1e10].groupby("Station")["Nitrate[µmol/kg]"].mean().plot(ax=axes[0])
    plt.ylabel("Nitrate[µmol/kg]")

    nitrateQF = df.columns.get_loc("Nitrate[µmol/kg]") + 1
    df[(df["Nitrate[µmol/kg]"] > -1e10) & (df.iloc[:, nitrateQF] == "0")].groupby("Station")[
        "Nitrate[µmol/kg]"].mean().plot(ax=axes[1])

    axes[0].set_ylabel("Nitrate[µmol/kg]")
    axes[0].set_title("Nitrate Data Profile, Including all Data")
    axes[1].set_title("Nitrate Data Profile, Only QF 0 Data")
    axes[1].yaxis.set_tick_params(labelbottom=True)
    plt.show()


#Depth[m]
#Temperature[°C]
def answerQ2(df: pd.DataFrame):
    tempQF = df.columns.get_loc("Temperature[°C]") + 1
    depthQF = df.columns.get_loc("Depth[m]") + 1
    df["Depth[m]"] = pd.to_numeric(df["Depth[m]"])
    df["Temperature[°C]"] = pd.to_numeric(df["Temperature[°C]"])

    binMap = np.arange(0, 1001, 5)
    df['bins'] = pd.cut(df["Depth[m]"], binMap, labels=binMap[:-1])

    # realValues = values = pd.Series(df[(df.iloc[:,depthQF] == "0") & (df.iloc[:,tempQF] == "0") & (df["Temperature[°C]"] > -1e10)].groupby(df["bins"])["Temperature[°C]"].mean().dropna())
    values = pd.Series(
        df[(df.iloc[:, depthQF] == "0") & (df.iloc[:, tempQF] == "0") & (df["Temperature[°C]"] > -1e10)].groupby(
            df["bins"])["Temperature[°C]"].mean().interpolate(method="linear"))
    std = values.std()
    mean = values.mean()

    plt.plot(values.index, values.array)
    plt.gca().set_xlabel("Depth[m]")
    plt.gca().set_ylabel("Temp")
    plt.gca().fill_between(values.index, min(values.array[1:]), max(values.array[1:]), alpha=.5,
                           where=(values.array < mean - std or values.array > mean + std))
    plt.title("Average Temperature at Depth, +-1 std shaded in blue ")
    plt.show()


def answerQ3(df: pd.DataFrame):
    depthQF = df.columns.get_loc("Depth[m]") + 1
    oxyQF = df.columns.get_loc("Oxygen[µmol/kg]") + 1
    df["Station"] = pd.to_numeric(df["Station"])
    df["Depth[m]"] = pd.to_numeric(df["Depth[m]"])
    df["Oxygen[µmol/kg]"] = pd.to_numeric(df["Oxygen[µmol/kg]"])
    oxyDF = df[(df.iloc[:, depthQF] == "0") & (df.iloc[:, oxyQF] == "0") & (df["Depth[m]"] > -1e10)
               & (df["Oxygen[µmol/kg]"] > -1e10) & (df["Depth[m]"] < 1001)][["Station", "Depth[m]", "Oxygen[µmol/kg]"]]

    contour = plt.tricontourf(oxyDF["Station"], oxyDF["Depth[m]"], oxyDF["Oxygen[µmol/kg]"])
    plt.gca().invert_yaxis()
    plt.gca().set_xlabel("Station #")
    plt.gca().set_ylabel("Depth[m]")
    plt.colorbar(contour).set_label("Oxygen[µmol/kg]")
    plt.title("Dissolved Oxygen at Depth Across All Stations")
    plt.show()


def argo1Helper(s: str, df: pd.DataFrame):
    def getSensors():
        pList = argoPList(df)
        sensors = []
        for p in pList:
            if s in p:
                sensors.append(p)
        return sensors

    def floatHasSensor(sensors, floatSensors):
        return any(sensor in floatSensors for sensor in sensors)

    sensors = getSensors()
    return len(df[df["parameters"].apply(lambda x: floatHasSensor(sensors, x))])


def argo1(df: pd.DataFrame):
    print("DOXY: " + str(argo1Helper("DOXY", df)))
    print("NITRATE: " + str(argo1Helper("NITRATE", df)))
    print("PH_IN_SITU: " + str(argo1Helper("PH_IN_SITU", df)))
    print("CHLA: " + str(argo1Helper("CHLA", df)))
    print("BBP: " + str(argo1Helper("BBP", df)))
    print("IRRADIANCE: " + str(argo1Helper("IRRADIANCE", df)))


def argo2(df: pd.DataFrame):
    timeSeries = pd.DataFrame(range(2000, 2023), columns=["date"])
    params = ["DOXY", "NITRATE", "PH_IN_SITU", "CHLA", "BBP"]

    for p in params:
        df[p] = df["parameters"].apply(lambda x: any(p in sensor for sensor in x))
        ts = \
        df[(df["date"].dt.year >= 2000) & (df["date"].dt.year <= 2023) & (df[p])].groupby(df["date"].dt.year).count()[p]
        timeSeries = timeSeries.merge(ts, left_on="date", right_on="date", how="outer")

    timeSeries.plot(x="date")
    plt.xticks(timeSeries["date"])
    plt.ylabel("Profiles per year")
    plt.xlabel("Year")
    plt.grid(True)
    plt.title("Number of Deployments per year, Grouped by Sensor Type")
    plt.show()


#come back to later
def argo3(df: pd.DataFrame):
    df["latitude"] = pd.to_numeric(df["latitude"])
    df["longitude"] = pd.to_numeric(df["longitude"])
    #grouped = df[df["date"].dt.year == 2023].groupby("institution")
    worldmap = gpd.read_file(geodatasets.get_path("naturalearth.land"))
    worldmap.plot(color = "grey")
    sns.scatterplot(data=df[df["date"].dt.year == 2023], x='longitude', y='latitude', hue="institution", s=20)
    plt.title("Map of 2023 Float Deployments, Grouped by Deploying Institution")
    plt.show()


def challengeQ():
    class challengeData:
        def __init__(self, floatID=None, actP=None, data=None, fixLoc=None, sourceFile=None):
            self.FloatID = floatID
            self.actP = actP
            self.data = data
            self.fixLoc = fixLoc
            self.fileName = None if sourceFile is None else sourceFile
            if sourceFile is not None:
                with open(sourceFile, encoding="UTF-8") as f:
                    readingData = False
                    samplesLeft = None
                    for line in f:
                        if readingData:
                            if line[0] == '$':
                                self.data = [str.split(line)[1:]]
                            elif line[0] == '#':
                                readingData = False
                            else:
                                if self.data is None:
                                    temp = str.split(line)
                                    if '(' in temp[-1]:
                                        self.data = [str.split(line[:-1])]
                                        if samplesLeft is not None:
                                            samplesLeft -= 1
                                    else:
                                        self.data = [str.split(line)]
                                        if samplesLeft is not None:
                                            samplesLeft -= 1
                                else:
                                    if len(line.split()) == len(self.data[0]):
                                        self.data.append(line.split())
                                    else:
                                        temp = [None] * len(self.data[0])
                                        for i in range(len(temp)):
                                            temp[i] = line.split()[i]
                                        self.data.append(temp)
                                    samplesLeft -= 1
                            if samplesLeft is not None and samplesLeft == 0:
                                readingData = False
                        else:
                            if self.FloatID is None:
                                if line.find("FloatId") > -1:
                                    self.FloatID = int(''.join(filter(str.isdigit, line)))
                            if self.actP is None:
                                if line.find("CpActivationP") > -1:
                                    self.actP = int(''.join(filter(str.isdigit, line)))
                            if self.fixLoc is None:
                                if line.find("IridiumFix") > -1:
                                    splt = str.split(line)
                                    for i in range(len(splt)):
                                        if splt[i].isnumeric() or splt[i][0] == '-':
                                            try:
                                                self.fixLoc = (float(splt[i]), float(splt[i + 1]))
                                            except:
                                                try:
                                                    self.fixLoc = (float(splt[i]), None)
                                                    print("ERROR Determining Latitude")
                                                except:
                                                    print("ERROR Determining Longitude and Latitude")
                                            break
                            if self.data is None:
                                if line.find("Discrete samples") > -1:
                                    readingData = True
                                    try:
                                        samplesLeft = int(str.split(line)[3])
                                    except:
                                        print("ERROR number of smamples could not be located")

        def showAtributes(self):
            print("Source File: " +str(self.fileName))
            print("FloatID: " + str(self.FloatID))
            print("CP Activation Pressure: " + str(self.actP))
            print("Profile Fix Location: " + str(self.fixLoc))
            if self.data is None:
                print("No Data Extracted from File")
            else:
                print("Data Successfully Extracted from File")

    cd1 = challengeData(sourceFile="19806.017.msg")
    cd2 = challengeData(sourceFile="19806.010.msg")
    cd2.showAtributes()
    print()
    cd1.showAtributes()


#pd.set_option('display.max_columns', 15)
#FV = parseVIZ("5904859QC.TXT")
#AD = parseARGO("argo_synthetic-profile_index.txt")

challengeQ()
