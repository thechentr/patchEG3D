import requests  

def download_file(url, output_path):  
    try:  
        response = requests.get(url, stream=True)  
        response.raise_for_status()

        with open(output_path, 'wb') as file:  
            for chunk in response.iter_content(chunk_size=8192):  
                file.write(chunk)  

        print(f"File downloaded successfully and saved as {output_path}.")  
    except requests.exceptions.HTTPError as http_err:  
        print(f"HTTP error occurred: {http_err}")  
    except Exception as err:  
        print(f"An error occurred: {err}")  


# see detail at https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/eg3d
url = "https://api.ngc.nvidia.com/v2/models/nvidia/research/eg3d/versions/1/zip"  
output_path = "checkpoints/eg3d_checkpoint.zip"  
download_file(url, output_path)