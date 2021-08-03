import os
from tools import STLUtils
import pandas as pd
pd.set_option('display.max_columns', None)


def getSTLs():
    cwd = os.getcwd()
    onlyfiles = [os.path.join(cwd, f) for f in os.listdir(cwd) if os.path.isfile(os.path.join(cwd, f))]
    paths = [i for i in onlyfiles if ".stl" in i or ".STL" in i]
    names = [i.split('\\')[-1] for i in paths]

    #print(names)
    #print(paths)

    return names, paths


def yesNo(input):
    d = {
        'y':True,
        'n':False,
        "":False
    }

    if input not in ["y","n", ""]:
        raise ValueError("Invalid input. Please specify 'y' or 'n'")

    return d[input]


def main():

    # Initialize STLUtility Class
    stlutils = STLUtils()

    # Initialize Results Data Structure
    table = {}

    # Detect files in folder
    names, paths = getSTLs()
    files = {i:j for i,j in zip(names,paths)}

    # Get input
    print("\n\n*** DM Quoting Tool ***\n")

    materials = ['316L', '17-4PH', '4140', 'H13']
    print("Select Material:")
    material = int(input("(1) 316L, (2) 17-4 PH, (3) 4140, (4) H13: "))
    material = materials[material]

    single_cycle = input("\nQuote single cycle? This will assume all parts under a single cycle; requires \n"
                         "inputting custom part quantities. (y/n): ")
    print(single_cycle)
    single_cycle = yesNo(single_cycle)


    #if single_cycle:
    #    input_qtys = input("\nSpecify individual part quantities? (y/n): ")
    #    input_qtys = yesNo(input_qtys)
    #else:
    #    input_qtys = False

    if single_cycle:#input_qtys:
        print("\nSpecify quantities for {0} STLs".format(len(names)))
        custom_qtys = {}
        for name in names:
            qty = int(input("{0}: ".format(name)))
            custom_qtys.update({name:qty})

    # Calculate Model Volumes
    print("\n\nCalculating model volumes.")
    volumes = {}
    boundingVolumes = {}
    for name,path in zip(names,paths):
        #print('Calculating model volume for {0}'.format(name))
        vol, bVol = stlutils.calculateVolume(path,material)
        volumes.update({name:vol})
        boundingVolumes.update({name: bVol})
        table.update({name:[vol,*bVol]})

    # Calculate Maximum Part Quantities for Each Step
    print('Calculating maximum part quantity per cycle.')
    quantities = {}
    for name in names:
        #print('Calculating quantities for {0}, with bv: {1}'.format(name,boundingVolumes[name]))
        qtys = stlutils.calculateQuantities(boundingVolumes[name],volumes[name],material)
        quantities.update({name:qtys})
        table[name] += [qtys['printer'],qtys['debinder'],qtys['furnace']]

    # Calculate Cycles
    print('Calculating cycles & costs.')
    cycles = {}
    cycleCosts = {}
    for name in names:
        #print('Calculating equipment cycle counts for {0}'.format(name))
        cycle,cost = stlutils.calculateCycles(quantities[name])
        cycles.update({name:cycle})
        cycleCosts.update({name: cost})
        table[name] += [cycle['printer'], cycle['debinder'], cycle['furnace']]
        table[name] += [cost['printer'], cost['debinder'], cost['furnace']]

    print("\n\n*** Calculations Complete ***\n\n")

    print('--- PART & CYCLE SUMMARY ---')
    # Formatting Data for Output
    headers = ['Volume (cm^3)', 'dx (mm)', 'dy (mm)', 'dz (mm)', 'perPrint', 'perDebind', 'perSinter', 'nPrints', 'nDebinds', 'nSinters', 'costPrint', 'costDebind', 'costSinter']
    table = pd.DataFrame.from_dict(table,orient='index')
    table.columns = headers

    # Unit Corrections
    table['Volume (cm^3)'] = table['Volume (cm^3)']/1000
    table.insert(1,'Material Unit Cost',table['Volume (cm^3)']*stlutils.materials[material]['cost'],allow_duplicates=True)
    table = table.round(2)

    # Output table
    print(table)
    table.to_csv('partSummary.csv')

    print('\n\n--- MANUFACTURING SUMMARY ---')
    summary = pd.DataFrame(index=table.index)
    if single_cycle:
        # Using Custom Quantities for Mfg. Summary
        summary['Quantity'] = pd.Series(custom_qtys)
    else:
        summary['Quantity'] = table['perSinter']
    summary['Total Material Cost'] = summary['Quantity']*table['Material Unit Cost']
    summary['Print Cost'] = [stlutils.equipment['printer']['cost']]*len(names)
    summary['Debind Cost'] = [stlutils.equipment['debinder']['cost']] * len(names)
    summary['Sinter Cost'] = [stlutils.equipment['furnace']['cost']] * len(names)
    summary['Cycle Cost'] = summary['Print Cost'] + summary['Debind Cost'] + summary['Sinter Cost']
    summary['Total Cost'] = summary['Cycle Cost'] + summary['Total Material Cost']
    summary['Unit Cost'] = summary['Total Cost']/summary['Quantity']

    print(summary)
    summary.to_csv('manufacturingSummary.csv')



if __name__ == "__main__":
    main()

