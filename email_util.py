# File: email_util.py

import logging
import smtplib
import threading
from app_secrets import FROM_EMAIL, SEND_EMAIL, APP_PASSWORD 
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from email.mime.text import MIMEText

logging.basicConfig(
    filename='debug.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
)


from threading import Lock

# Initialize the lock
email_send_lock = Lock()

def send_email(image_path):
    with email_send_lock:
        thread = threading.Thread(target=send_email_start, args=(image_path,))
        thread.start()
        thread.join() 

def send_email_start(image_path):
    print('Starting send_email')

    try: 
        msg = MIMEMultipart()
    
        msg['From'] = FROM_EMAIL
        msg['To'] = SEND_EMAIL
        msg['Subject'] = "Uncompliance Detected!!!"
    
        body = "This Guy Need to Go Jail!!!"
    
        msg.attach(MIMEText(body, 'plain'))
    
        filename = "detected_frame.jpg"
        attachment = open(image_path, "rb")
    
        part = MIMEBase('application', 'octet-stream')
        part.set_payload((attachment).read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', "attachment; filename= %s" % filename)
    
        msg.attach(part)
    
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(FROM_EMAIL, APP_PASSWORD)
        text = msg.as_string()
        server.sendmail(FROM_EMAIL, SEND_EMAIL, text)
        server.quit()
        logging.info('Email sent successfully.')
    except Exception as e:
        print(f'Failed to send email. Reason: {e}')
        
    print('Finished send_email function.')

