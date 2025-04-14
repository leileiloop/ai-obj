import smtplib
import mysql.connector
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# SMTP Server Configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587  # Use 465 for SSL, 587 for TLS
EMAIL_ADDRESS = "2020-200423@rtu.edu.ph"
EMAIL_PASSWORD = "otfu klum scxj akhz"  # Use App Password instead of your actual password

# Connect to MySQL Database
db = mysql.connector.connect(
    host="127.0.0.1",      # Change this if your database is hosted remotely
    user="root",   # Your MySQL username
    password="",  # Your MySQL password
    database="test"        # Your database name
)
cursor = db.cursor()

# Fetch Emails from Database
cursor.execute("SELECT Email FROM users")
emails = cursor.fetchall()  # List of tuples (each row is a tuple)

for email_tuple in emails:
    recipient_email = email_tuple[0]  # Extract email from tuple

    # Create Email
    msg = MIMEMultipart()
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = recipient_email
    msg["Subject"] = "Test Email from Python"

    body = f"Hello {recipient_email},\nGragraduate na."
    msg.attach(MIMEText(body, "plain"))

    try:
        # Connect to SMTP Server
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()  # Secure the connection
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)

        # Send Email
        server.sendmail(EMAIL_ADDRESS, recipient_email, msg.as_string())
        print(f"Email sent successfully to {recipient_email}")

        # Close the connection
        server.quit()
    except Exception as e:
        print(f"Error sending email to {recipient_email}: {e}")

# Close Database Connection
cursor.close()
db.close()
