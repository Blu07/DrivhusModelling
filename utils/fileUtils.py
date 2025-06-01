from json import dump, load


def saveParamsToFile(content, file_name: str = "u_values.json"):
    """ Save the parameters for each day to a JSON file """    
    
    # Open the file in read mode to get the object
    # Change the objects value
    # Write the updated object back to the file
    

    data = loadParamsFromFile(file_name)
    data.update(content)
    
    try: 
        with open(file_name, "w") as f:
            dump(data, f, indent=4)
            
    except Exception as e:
        print(f"Kunne ikke lagre verdier: {e}")
        return False
    
    return True


def loadParamsFromFile(file_name: str = "u_values.json"):
    """ Load the parameters for each day from a JSON file """
    
    try:
        with open(file_name, "r") as f:
            return load(f)
    except FileNotFoundError:
        print(f"Finner ikke filen {file_name}.")
        return {}
    except Exception as e:
        print(f"Kunne ikke lese lagrede verdier: {e}")
        return {}


