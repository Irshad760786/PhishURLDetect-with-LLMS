# PhishURLDetect-with-LLMS
# Dataset
The paper used ~0.6M URLs (178,142 phishing
URLs and 395,738 benign URLs.). There are few places to gather malicious URLs. My recommendation is to do the following:

Phishing URLs
OpenPhish will provide 500 malicious URLs for free in TXT form. You can access that data here.

Likewise, PhishTank is an excellent resource that provides a daily feed of malicious URLs in CSV or JSON format. You can gather ~5K through the following link.

Finally, there is an excellent OpenSource project, Phishing.Database, run by Mitchell Krog. There is a ton of data available here to plus up your dataset.

Benign Data
I gathered benign URL data via two methods. The first was to use the top 50K domains from Alexa.

Next I used my own Chrome browser history to get an additional 60K. It was pretty easy to do on my Macbook. First, make sure your browser is closed. Then in your terminal run the following command:
