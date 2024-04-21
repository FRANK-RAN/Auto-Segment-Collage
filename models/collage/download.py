import gdown

url = 'https://drive.google.com/uc?id=1CsUNWLHCciq_0QlHmGoeGAYRcyzavB3z'
output = 'AIC.zip'

gdown.download(url, output, quiet=False)