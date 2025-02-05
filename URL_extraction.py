# Importing required packages
import pandas as pd
import os
import requests
import whois
from urllib.parse import urlparse
import re
from concurrent.futures import ThreadPoolExecutor, as_completed  # For parallel processing
from tqdm import tqdm  # For progress tracking

# Function to preprocess URLs
def preprocess_url(url):
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url  # Assume HTTP if no protocol is specified
    return url

# Function to extract domain from URL
def getDomain(url):
    try:
        domain = urlparse(url).netloc
        return domain
    except:
        return ""

# Function to check if IP address is present in URL
def havingIP(url):
    try:
        if re.search(r'\d+\.\d+\.\d+\.\d+', url):
            return 1
        else:
            return 0
    except:
        return 0

# Function to check if '@' symbol is present in URL
def haveAtSign(url):
    return 1 if '@' in url else 0

# Function to get the length of the URL
def getLength(url):
    return 1 if len(url) >= 54 else 0

# Function to get the depth of the URL
def getDepth(url):
    try:
        path = urlparse(url).path
        return len([i for i in path.split('/') if i])
    except:
        return 0

# Function to check for redirection '//' in URL
def redirection(url):
    pos = url.rfind('//')
    return 1 if pos > 6 else 0

# Function to check if 'http' or 'https' is in the domain part of the URL
def httpDomain(url):
    domain = urlparse(url).netloc
    return 1 if 'http' in domain else 0

# Function to check for URL shortening services
def tinyURL(url):
    shortening_services = r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|db\.tt|qr\.ae|adf\.ly|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|q\.gs|is\.gd|po\.st|bc\.vc|twit\.this|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|tr\.im|link\.zip\.net"
    match = re.search(shortening_services, url)
    return 1 if match else 0

# Function to check for prefix or suffix '-' in domain
def prefixSuffix(url):
    domain = urlparse(url).netloc
    return 1 if '-' in domain else 0

# Function to check for DNS record availability
def dnsRecord(domain):
    try:
        whois.whois(domain)
        return 0
    except:
        return 1

# Function to check web traffic ranking
def web_traffic(url):
    try:
        alexa_rank = requests.get(f"http://data.alexa.com/data?cli=10&url={url}", timeout=3)
        rank = int(re.search(r'<POPULARITY[^>]*TEXT="(\d+)"', alexa_rank.text).group(1))
        return 1 if rank < 100000 else 0
    except:
        return 1

# Function to check domain age
def domainAge(domain_name):
    try:
        creation_date = domain_name.creation_date
        expiration_date = domain_name.expiration_date
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        if isinstance(expiration_date, list):
            expiration_date = expiration_date[0]
        age = (expiration_date - creation_date).days / 30
        return 1 if age >= 12 else 0
    except:
        return 1

# Function to check domain end period
def domainEnd(domain_name):
    try:
        expiration_date = domain_name.expiration_date
        if isinstance(expiration_date, list):
            expiration_date = expiration_date[0]
        end = (expiration_date - pd.Timestamp.now()).days / 30
        return 1 if end >= 6 else 0
    except:
        return 1

# Function to check for IFrame redirection
def iframe(response):
    if not response or response == "":
        return 1
    try:
        if re.findall(r"[<iframe>|<frameBorder>]", response.text):
            return 0
        else:
            return 1
    except AttributeError:
        return 1

# Function to check for mouse over events
def mouseOver(response):
    if not response or response == "":
        return 1
    try:
        if re.findall("<script>.+onmouseover.+</script>", response.text):
            return 1
        else:
            return 0
    except AttributeError:
        return 0

# Function to check for right-click disabling
def rightClick(response):
    if not response or response == "":
        return 1
    try:
        if re.findall(r"event.button ?== ?2", response.text):
            return 0
        else:
            return 1
    except AttributeError:
        return 1

# Function to check for website forwarding
def forwarding(response):
    if not response or response == "":
        return 1
    try:
        if len(response.history) <= 2:
            return 0
        else:
            return 1
    except AttributeError:
        return 1

# Function to extract features from URLs
def featureExtraction(url, label):
    features = []
    url = preprocess_url(url)

    # Address bar features
    features.append(getDomain(url))
    features.append(havingIP(url))
    features.append(haveAtSign(url))
    features.append(getLength(url))
    features.append(getDepth(url))
    features.append(redirection(url))
    features.append(httpDomain(url))
    features.append(tinyURL(url))
    features.append(prefixSuffix(url))

    # Domain-based features
    dns = 0
    domain_name = None
    try:
        domain = urlparse(url).netloc
        domain_name = whois.whois(domain)
    except Exception as e:
        dns = 1

    features.append(dns)
    features.append(web_traffic(url))
    features.append(1 if dns == 1 else domainAge(domain_name))
    features.append(1 if dns == 1 else domainEnd(domain_name))

    # HTML & JS features
    response = None
    try:
        response = requests.get(url, timeout=3)
    except (requests.exceptions.RequestException, requests.exceptions.Timeout):
        response = None  # Skip if it takes too long

    features.append(iframe(response))
    features.append(mouseOver(response))
    features.append(rightClick(response))
    features.append(forwarding(response))
    features.append(label)

    return features

# Function to load dataset and identify URL column
def load_dataset(file_path, url_column=None):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist. Please ensure it is in the correct directory.")

    data = pd.read_csv(file_path)

    if data.empty:
        raise ValueError(f"The dataset {file_path} is empty. Please provide a valid dataset.")

    # Display columns for debugging
    print(f"Columns in {file_path}: {data.columns.tolist()}")  

    # Manually select the URL column if not provided
    if not url_column:
        url_column = data.columns[0]  # Default to the first column if none specified

    if url_column not in data.columns:
        raise KeyError(f"The specified URL column '{url_column}' does not exist in the dataset {file_path}. Please verify the dataset structure.")

    return data, url_column

# Load Phishing URLs Data
phishing_data_path = 'DataFiles/2.online-valid.csv'
phish_data, phish_url_column = load_dataset(phishing_data_path, url_column='https://www.southbankmosaics.com')

# Load Legitimate URLs Data
legitimate_data_path = 'DataFiles/1.Benign_list_big_final.csv'
legit_data, legit_url_column = load_dataset(legitimate_data_path)

# Sample 5,000 phishing and legitimate URLs
phish_sample = phish_data.sample(n=5000, random_state=12).reset_index(drop=True)
legit_sample = legit_data.sample(n=5000, random_state=12).reset_index(drop=True)

# Function to extract features in parallel
def parallel_feature_extraction(url_list, label):
    features = []
    with ThreadPoolExecutor(max_workers=20) as executor:  # Adjust max_workers as needed
        futures = {executor.submit(featureExtraction, url, label): url for url in url_list}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting features"):
            try:
                result = future.result()
                features.append(result)
            except Exception as e:
                print(f"Error processing URL: {futures[future]} - {e}")
    return features

# Extract features from phishing URLs in parallel
print("Extracting features from phishing URLs...")
phish_features = parallel_feature_extraction(phish_sample[phish_url_column], label=1)

# Extract features from legitimate URLs in parallel
print("Extracting features from legitimate URLs...")
legit_features = parallel_feature_extraction(legit_sample[legit_url_column], label=0)

# Combine phishing and legitimate features into one dataset
final_dataset = phish_features + legit_features

# Create DataFrame and save to CSV
columns = ['Domain', 'Having_IP', 'Have_At_Sign', 'URL_Length', 'URL_Depth', 'Redirection', 
           'HTTP_in_Domain', 'TinyURL', 'Prefix_Suffix', 'DNS_Record', 'Web_Traffic', 
           'Domain_Age', 'Domain_End', 'Iframe', 'Mouse_Over', 'Right_Click', 'Web_Forwards', 'Label']

final_df = pd.DataFrame(final_dataset, columns=columns)
final_df.to_csv('final_dataset.csv', index=False)

print("Final dataset created and saved successfully.")
