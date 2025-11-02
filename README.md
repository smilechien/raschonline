 
  # RaschOnline
*A Rasch Rating Scale Modeling Software Using the Copy-and-Paste Approach*

---

## 🧩 Overview
**RaschOnline** is a web-based application for performing **Rasch Rating Scale Modeling (RSM)** directly in a browser.  
It allows users to copy-and-paste data into an input box, execute Rasch model analysis, and visualize item–person statistics such as Wright maps, KIDMAPs, and fit plots.  
The system is implemented in **Classic ASP** and designed for educational and clinical psychometrics.

---

## ⚙️ Features
- Copy-and-paste data entry (no file upload required)  
- Automatic detection of item and person fields  
- Rasch Rating Scale Model computation  
- Output of item/person fit, reliability, and Wright/KIDMAP graphs  
- Classic ASP backend (no installation of R or external packages required)  
- Runs locally under IIS

---

## 🧠 System Requirements
| Component | Version / Note |
|------------|----------------|
| Windows OS | Windows 10 or higher |
| Web Server | IIS (Internet Information Services) |
| Browser | Any modern browser (Chrome, Edge, Firefox) |
| Backend | Classic ASP (VBScript) |
| Optional | MS Access or SQL database for data storage |

---

## 🚀 Installation
1. Clone or download this repository:
   ```bash
   git clone https://github.com/yourusername/RaschOnline.git
Copy the folder raschonline to your IIS root directory:

 
C:\inetpub\wwwroot\raschonline\
Enable ASP under Windows Features → IIS → Application Development.

Start IIS (iisreset if needed).

Open the application in your browser:

classic asp code:
http://localhost/raschonline/raschonline.asp
💡 Usage
Open the web page raschonline.asp.

Copy and paste your dataset into the input field (sample data are provided).

Choose model options (Person Fit, KIDMAP, ICC, etc.).

Click Run Rasch Model.

View the generated statistics and plots.

🧮 Example Dataset
 
Item1 Item2 Item3 Item4 Item5 Item6 Item7 Item8 Item9 Item10 name group
1 1 1 1 1 1 1 0 1 Student1 1
1 1 1 1 1 1 1 1 0 Student2 1
1 1 1 1 1 1 1 0 1 Student3 1
0 1 1 1 1 1 1 0 1 Student4 0
📁 File Structure
 
RaschOnline/
│
├── raschonline.asp        # Main ASP program
├── raschrsm.asp           # Supporting module
├── LICENSE                # License file
├── README.md              # This documentation
└── /autoadjust/js/        # JavaScript utilities
🧾 Citation
If you use this software, please cite:

Chien TW. RaschOnline: A Rasch Rating Scale Modeling Software Using the Copy-and-Paste Approach.
SoftwareX, 2025, V2.0. https:/www.raschonline.com/

📜 License
This project is licensed under the MIT License.
See the LICENSE file for details.

🙌 Acknowledgements
This work was developed by Tsair-Wei Chien to simplify Rasch model applications in education and health research.
