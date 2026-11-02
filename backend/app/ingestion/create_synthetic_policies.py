from html import escape
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "data" / "expanded_raw"

# All policies below are fictional and created for RAG portfolio testing.
POLICIES = {
    "finance_reimbursement_policy.html": {
        "title": "Finance and Reimbursement Policy",
        "owner": "Finance Department",
        "sections": {
            "Expense eligibility": [
                "Employees may claim reasonable business expenses incurred while performing approved company duties. Personal purchases, entertainment unrelated to business, and expenses incurred for family members are not eligible.",
                "Every expense claim must identify the business purpose, transaction date, amount, currency, department, and project where applicable. Missing information can delay review.",
                "Expenses must be submitted through the Finance Portal. Email submissions are accepted only during a documented portal outage and must be entered into the portal after service is restored.",
                "Employees must not divide one purchase into multiple transactions to avoid an approval threshold. Finance may combine related purchases when determining the required approval level."
            ],
            "Approval limits": [
                "Expenses up to 500 US dollars require line manager approval. Expenses above 500 and up to 2,500 dollars require Department Director approval.",
                "Expenses above 2,500 and up to 10,000 dollars require Finance Director approval before the commitment is made. Expenses exceeding 10,000 dollars require Chief Financial Officer approval.",
                "Recurring subscriptions are evaluated using their total expected twelve month cost rather than a single monthly payment.",
                "An employee cannot approve their own reimbursement. When the employee's manager submitted the expense, approval must be escalated to the next management level."
            ],
            "Submission deadlines": [
                "Employees should submit expense reports within 30 calendar days after the transaction date or completion of the business trip, whichever occurs later.",
                "Reports submitted between 31 and 60 calendar days require a written explanation and manager approval. Finance may reject claims submitted more than 60 calendar days after the expense.",
                "A report containing several transactions should include a separate line for each expense. Combining unrelated transactions into one unexplained amount is not permitted.",
                "Employees who discover an incorrect claim must notify Finance within five business days of discovering the error and provide corrected supporting information."
            ],
            "Receipts and evidence": [
                "An itemized receipt is required for individual expenses of 25 dollars or more. A card statement alone does not replace an itemized receipt.",
                "For expenses below 25 dollars, employees must provide a transaction description, date, amount, and business purpose even when a receipt is not available.",
                "For lost receipts, employees must complete a missing receipt declaration and obtain manager approval. Repeated missing receipt declarations may be reviewed by Finance.",
                "Foreign currency expenses must include the original currency amount and the conversion rate or card settlement amount used for reimbursement."
            ],
            "Payment and corrections": [
                "Approved reimbursement is normally included in the next scheduled payment cycle, which runs twice per month. Incomplete bank details can delay payment.",
                "Payments are transferred to the bank account registered in the employee payroll profile. Finance does not send reimbursements to personal accounts supplied only in an email.",
                "Duplicate claims must be reported immediately. Any duplicate payment must be returned or offset against a future reimbursement in accordance with Finance instructions.",
                "Employees may request a status update through the Finance Portal after five business days without a status change."
            ],
            "Travel and special expenses": [
                "Travel expenses must also comply with the separate Travel and Expense Policy. A reimbursement approval does not replace advance travel authorization.",
                "Client meals require the business purpose, attendee names, organization names, and itemized receipt. Alcohol is not reimbursable under this policy.",
                "Training expenses require proof of enrollment or attendance. Conference registration must be approved before payment when the cost exceeds the employee's delegated spending limit.",
                "Finance may request additional evidence where an expense appears unusual, duplicated, incorrectly categorized, or inconsistent with the approved business purpose."
            ]
        }
    },
    "procurement_policy.html": {
        "title": "Procurement and Purchasing Policy",
        "owner": "Procurement Department",
        "sections": {
            "Purchase planning": [
                "Departments must define the business need, expected quantity, delivery location, required date, and estimated total cost before requesting a purchase.",
                "A purchase request must be created before a purchase order or supplier commitment is made. Employees must not instruct vendors to begin work before authorization.",
                "The requester must check whether an approved internal supplier or existing contract can satisfy the requirement before sourcing a new vendor.",
                "Artificially splitting purchases to bypass approval limits is prohibited. Related purchases are assessed using their combined expected value."
            ],
            "Delegated approval": [
                "Purchases up to 1,000 dollars require department manager approval. Purchases above 1,000 and up to 5,000 dollars require Department Director approval.",
                "Purchases above 5,000 and up to 20,000 dollars require Procurement Director approval. Purchases exceeding 20,000 dollars require Chief Financial Officer approval.",
                "Software purchases that process company information require IT Security review regardless of purchase value.",
                "Equipment purchases above 10,000 dollars require asset management review before a purchase order is issued."
            ],
            "Competitive quotations": [
                "Purchases from 1,000 to 5,000 dollars should normally include two comparable vendor quotations. The requester must document the comparison.",
                "Purchases above 5,000 dollars require at least three written quotations unless an approved exception applies.",
                "A single-source purchase requires a written justification explaining compatibility, intellectual property, urgency, or the absence of reasonable alternatives.",
                "Quotations must describe equivalent specifications, quantities, service periods, taxes, delivery costs, and warranty terms before they are compared."
            ],
            "Purchase orders": [
                "Only Procurement may issue an official company purchase order. A purchase request or manager email is not a substitute for an authorized purchase order.",
                "The purchase order must state the supplier, approved amount, delivery address, payment terms, item description, and authorized requester.",
                "Changes to price, quantity, scope, or delivery terms must be reviewed before the supplier performs the additional work.",
                "The employee receiving goods must record the delivery date, quantity received, visible damage, and any difference from the purchase order."
            ],
            "Invoice matching": [
                "Accounts Payable checks invoices against the purchase order and receiving confirmation before releasing payment.",
                "An invoice with a price difference greater than five percent requires Procurement review and written confirmation from the budget owner.",
                "Invoices without a valid purchase order must be returned for correction unless Procurement approves a documented exception.",
                "Suppliers must send invoices to the company Accounts Payable address rather than directly to individual employees."
            ],
            "Supplier conflicts and records": [
                "Employees must disclose personal, financial, or family relationships with a supplier before participating in vendor selection.",
                "Procurement evaluations must use documented business criteria such as total cost, delivery capability, service quality, security, and contract terms.",
                "Bid documents, quotations, approval records, purchase orders, and receiving records must be retained for at least seven years.",
                "Employees must not accept personal commissions, cash gifts, or undisclosed benefits from vendors involved in a purchasing decision."
            ]
        }
    },
    "remote_work_policy.html": {
        "title": "Remote Work and Flexible Working Policy",
        "owner": "Human Resources Department",
        "sections": {
            "Eligibility": [
                "Employees who have completed their initial 90 day probation period may request a regular remote work arrangement, subject to role requirements and manager approval.",
                "Roles that require physical equipment handling, confidential paper records, or continuous on-site customer service may have limited remote eligibility.",
                "Employees must have a suitable working environment, reliable internet access, and the equipment required for their assigned responsibilities.",
                "A remote work arrangement is a work-location agreement and does not change the employee's contractual working hours or performance expectations."
            ],
            "Work schedules": [
                "Employees on a hybrid schedule should record their planned office and remote days in the team calendar by the final working day of the preceding week.",
                "Core collaboration hours are 10 AM to 4 PM in the employee's approved work location time zone unless the team has a documented alternative schedule.",
                "Employees must obtain manager approval before changing a fixed on-site day when the change affects meetings, customer support, or team coverage.",
                "Remote employees must remain reachable through approved communication tools during their scheduled working hours and update their availability when stepping away."
            ],
            "Remote work requests": [
                "A regular remote arrangement requires a written request to the line manager stating the proposed schedule, start date, work location, and reason for the arrangement.",
                "The manager should respond within five business days after receiving a complete request and may consult HR about role suitability.",
                "Temporary remote work outside the normal schedule may be approved for up to five consecutive working days by the line manager.",
                "Arrangements lasting longer than 30 calendar days must be reviewed by HR and the Department Director."
            ],
            "Information security": [
                "Company work must be performed using approved devices, applications, and authentication methods. Personal email accounts must not be used to transfer company documents.",
                "Employees must use the company approved VPN when accessing internal systems from an untrusted network or public Wi-Fi.",
                "Screens must be locked whenever unattended. Printed company documents must be stored securely and must not be left in public locations.",
                "Confidential calls must be conducted where conversations cannot reasonably be overheard by unauthorized people."
            ],
            "Equipment and expenses": [
                "The company may issue a laptop, headset, keyboard, and security accessories based on role requirements and manager authorization.",
                "Employees must report damaged, lost, or stolen company equipment to IT Support within four hours of discovering the incident.",
                "Home internet and electricity expenses are not automatically reimbursable. Any approved remote work allowance must be documented by HR.",
                "Employees may not connect unapproved storage devices or allow family members to use company-managed devices."
            ],
            "Review and termination": [
                "Managers may review remote work arrangements every six months based on role suitability, performance, attendance, and business requirements.",
                "Repeated failure to attend required office days without approval may result in modification or withdrawal of the remote work arrangement.",
                "The company may require temporary on-site attendance for audits, training, incident response, or important customer activities.",
                "Employees changing their home work location must notify HR and IT before working from the new location."
            ]
        }
    },
    "data_privacy_policy.html": {
        "title": "Data Privacy and Information Handling Policy",
        "owner": "Information Security and Privacy Office",
        "sections": {
            "Data classification": [
                "Company information is classified as Public, Internal, Confidential, or Restricted according to the harm that could result from unauthorized access or disclosure.",
                "Public information may be shared externally after approval by the responsible content owner. Internal information is intended for employees and approved contractors.",
                "Confidential information includes non-public financial records, customer information, employee records, and commercial plans.",
                "Restricted information includes credentials, authentication secrets, highly sensitive personal data, and security investigation material."
            ],
            "Access and use": [
                "Employees may access personal data only when it is necessary for an approved business purpose and permitted by their assigned role.",
                "Access must be granted using individual accounts. Shared user accounts are prohibited unless a documented technical exception has been approved.",
                "Employees must not copy customer information into personal cloud storage, personal email, or unapproved AI services.",
                "Managers must request access removal when an employee changes role, leaves a project, or no longer needs the information."
            ],
            "Data retention": [
                "Customer support records must be retained for three years after the support case is closed unless a legal hold or contract requires a different period.",
                "Employee recruitment records for unsuccessful candidates must normally be retained for 12 months after the recruitment decision.",
                "Financial transaction records must be retained for seven years to support audits and statutory reporting requirements.",
                "Data must not be retained indefinitely without a documented business, contractual, or legal requirement."
            ],
            "Sharing and transfers": [
                "Confidential information may be shared with an external processor only after the supplier has passed the required security and privacy review.",
                "Transfers must use approved encrypted channels. Public file-sharing links must not be used for confidential or restricted information.",
                "An employee sending personal data to another department must verify the recipient's authorization and use the minimum information necessary.",
                "Cross-border transfers of personal data require Privacy Office approval and a review of applicable contractual and legal safeguards."
            ],
            "Privacy incidents": [
                "Suspected unauthorized disclosure, accidental email transmission, or loss of personal information must be reported to the Privacy Office within one hour of discovery.",
                "Employees must preserve relevant messages, file names, timestamps, and system details. They must not attempt to conceal or independently investigate a suspected breach.",
                "The incident response team determines whether affected records must be secured, access revoked, recipients contacted, or regulators notified.",
                "Employees must cooperate with incident investigations and provide accurate facts without modifying or deleting relevant evidence."
            ],
            "Data subject requests": [
                "Requests to access, correct, or delete personal information must be forwarded to the Privacy Office within one business day.",
                "Employees must not independently promise a deletion deadline or disclose personal data before the requester's identity and authority are verified.",
                "The Privacy Office maintains a request register containing the request date, request category, assigned owner, verification status, and response date.",
                "Exceptions to deletion may apply where records must be retained for legal, contractual, security, or financial purposes."
            ]
        }
    },
    "employee_benefits_policy.html": {
        "title": "Employee Benefits and Wellness Policy",
        "owner": "Human Resources Department",
        "sections": {
            "Medical coverage": [
                "Eligible full-time employees may enroll in the company group medical plan after completing 30 calendar days of employment.",
                "Employees may add an eligible spouse and dependent children during initial enrollment or an approved qualifying life event.",
                "The current plan provides an annual base coverage limit of 300,000 units of the plan currency for an enrolled employee, subject to insurer terms.",
                "Coverage details, network hospitals, exclusions, and claim procedures are provided in the insurer's separate plan documents."
            ],
            "Enrollment changes": [
                "New employees must submit benefit enrollment selections within 15 calendar days after their eligibility date.",
                "Changes following marriage, birth, adoption, or loss of other coverage must be submitted within 30 calendar days after the event.",
                "HR may request supporting documents before approving a dependent addition or change to the selected coverage.",
                "Employees must review dependent details and contact information annually during the open enrollment period."
            ],
            "Wellness support": [
                "Employees may access up to four confidential wellness counselling sessions per calendar year through the approved employee assistance service.",
                "Participation in wellness counselling is voluntary. Managers do not receive individual session notes or personal counselling details.",
                "The company organizes periodic health awareness activities and may offer preventive health screening based on the annual benefits plan.",
                "Urgent medical emergencies must be handled through local emergency services and must not be delayed while seeking HR approval."
            ],
            "Insurance claims": [
                "Employees should contact the insurer or designated benefits administrator to confirm whether a planned treatment is covered before a non-emergency admission.",
                "Cashless treatment is available only at participating network facilities and is subject to insurer authorization.",
                "Reimbursement claims must normally be submitted within 45 calendar days after discharge or completion of outpatient treatment.",
                "Employees must retain original bills, prescriptions, diagnostic reports, discharge summaries, and proof of payment when required by the insurer."
            ],
            "Benefit exclusions": [
                "Cosmetic procedures without medical necessity are excluded unless the plan document expressly provides coverage.",
                "Expenses incurred before the coverage effective date are not reimbursable under the group plan.",
                "An employee may not claim the same medical expense from two company benefit programs for reimbursement exceeding the actual expense.",
                "Coverage for pre-existing conditions, waiting periods, maternity expenses, and specific treatments is determined by the current insurer contract."
            ],
            "Benefit administration": [
                "Benefit questions should be submitted through the HR Benefits Portal with the employee identifier and a short description of the issue.",
                "HR should acknowledge a complete benefits query within three business days and provide an update when the insurer must be consulted.",
                "Employees leaving the company should review the insurer's coverage end date and any legally available continuation options.",
                "The company may change benefit providers or plan terms at renewal, subject to notice and applicable employment requirements."
            ]
        }
    },
    "hardware_equipment_policy.html": {
        "title": "Hardware and IT Equipment Policy",
        "owner": "Information Technology Department",
        "sections": {
            "Equipment allocation": [
                "Employees receive standard equipment according to their job role, approved workload, and available company inventory.",
                "A standard engineering workstation includes one managed laptop, one external monitor, a keyboard, a mouse, and an approved headset.",
                "Additional monitors, specialist devices, and higher-performance equipment require manager approval and confirmation from IT Asset Management.",
                "Company equipment remains company property and must be recorded against the employee's asset register."
            ],
            "Acceptable use": [
                "Company devices must be used primarily for authorized business activities and must comply with the IT Security Policy.",
                "Employees must not disable endpoint protection, disk encryption, operating system updates, or centrally managed security controls.",
                "Installing software with administrator privileges requires IT approval and a valid business justification.",
                "Employees must not lend company devices to family members, visitors, or unauthorized contractors."
            ],
            "Maintenance and repairs": [
                "Hardware defects must be reported through the IT Service Desk with the asset tag, device model, operating system, and a description of the problem.",
                "Employees must not open, repair, or replace internal device components unless specifically authorized by IT Support.",
                "Approved repair vendors must provide a work record containing the device serial number, repair description, parts replaced, and return date.",
                "IT Support should provide an initial response to a standard hardware ticket within one business day."
            ],
            "Loss and theft": [
                "Lost or stolen company devices must be reported to IT Security within one hour of discovery, even when the device is believed to be powered off.",
                "Employees must provide the asset tag, last known location, approximate incident time, and any available police report reference.",
                "IT Security may remotely lock or wipe a device when the loss creates a risk to company information.",
                "Employees must not attempt to recover a stolen device by confronting a suspected offender."
            ],
            "Temporary replacement": [
                "IT Support may issue a loan device when the primary device cannot be used for more than one business day and suitable stock is available.",
                "Loan devices must be assigned to an employee in the asset register and returned within three business days after replacement equipment is ready.",
                "Employees must not copy restricted information to a loan device unless it is managed and approved for that data classification.",
                "A returned damaged device must include the repair ticket reference and a description of any known incident."
            ],
            "Return and disposal": [
                "Employees must return all assigned devices, chargers, accessories, smart cards, and storage media on their final working day unless HR approves another date.",
                "IT Asset Management verifies the returned asset tag and records missing or damaged items before closing the employee's asset record.",
                "Devices approved for disposal must be securely erased using the company's approved sanitization procedure.",
                "No employee may sell, donate, dismantle, or transfer company hardware without written authorization from IT Asset Management."
            ]
        }
    },
    "learning_development_policy.html": {
        "title": "Learning and Professional Development Policy",
        "owner": "Learning and Development Department",
        "sections": {
            "Training eligibility": [
                "Regular full-time employees may request role-related external training after completing 90 calendar days of employment.",
                "Training must support the employee's current responsibilities, an agreed development objective, or a documented future role requirement.",
                "Mandatory compliance training is assigned separately and must be completed by the deadline shown in the learning platform.",
                "Temporary workers and contractors may participate when their contract and department manager authorize the training."
            ],
            "Annual learning budget": [
                "Each eligible employee may request up to 1,200 dollars in professional development funding per calendar year.",
                "Requests from 1,200 to 2,500 dollars require Department Director and Learning and Development approval as an exception.",
                "Amounts above 2,500 dollars require Human Resources Director approval and a documented strategic business justification.",
                "Unused annual learning funds expire at the end of the calendar year and cannot be transferred to another employee."
            ],
            "Approval process": [
                "Employees must submit a training request at least 15 business days before the course start date.",
                "The request must include the provider, course title, learning objectives, schedule, total cost, and expected work impact.",
                "Managers should respond within five business days. Lack of response does not constitute approval.",
                "Training must not be booked or paid for until the required approvals are recorded in the Learning Portal."
            ],
            "Attendance requirements": [
                "Employees receiving company-funded training must attend the required sessions and complete the provider's assessments when applicable.",
                "A completion certificate or attendance record must be uploaded within ten business days after the course ends.",
                "Employees who cannot attend must notify their manager and the training coordinator as soon as the conflict becomes known.",
                "Repeated non-attendance without an approved reason may affect eligibility for future discretionary development funding."
            ],
            "External certifications": [
                "Certification examination fees may be reimbursed after the employee submits proof of payment and the official examination result.",
                "A failed first attempt may be eligible for one additional attempt when the manager confirms that the certification remains relevant.",
                "Preparation materials must be purchased through approved channels and remain subject to the annual learning funding limit.",
                "Professional memberships may be funded when the membership directly supports the employee's responsibilities and has manager approval."
            ],
            "Knowledge sharing": [
                "Employees attending funded external training should share useful learning with their team within 30 calendar days after completion.",
                "Knowledge sharing may take the form of a short presentation, written summary, demonstration, or documented improvement proposal.",
                "Training notes must not reproduce licensed course materials or confidential information belonging to the provider.",
                "Managers should record agreed development outcomes during the next regular performance discussion."
            ]
        }
    },
    "recruitment_onboarding_policy.html": {
        "title": "Recruitment and Employee Onboarding Policy",
        "owner": "Talent Acquisition Department",
        "sections": {
            "Workforce requisitions": [
                "A department must obtain an approved workforce requisition before publishing a vacancy or engaging a recruitment agency.",
                "The requisition must state the role title, reporting manager, employment type, location, budget range, and business justification.",
                "Replacement positions must identify the departing position or approved vacancy reference.",
                "Changes to salary range, reporting structure, or headcount after approval require renewed authorization from the budget owner."
            ],
            "Candidate screening": [
                "Recruiters assess applicants against documented role criteria such as required skills, experience, education, and work authorization.",
                "Candidate evaluations must be based on job-related evidence and must not rely on protected personal characteristics.",
                "At least two authorized interviewers should participate in the final technical or functional assessment for regular full-time positions.",
                "Interview notes must record the questions asked, relevant evidence, assessment outcomes, and reasons for the hiring recommendation."
            ],
            "Offer approval": [
                "An employment offer must be approved by the hiring manager and the authorized HR compensation representative before it is sent.",
                "Offers above the approved salary range require Finance and HR Director approval.",
                "The written offer must state the job title, employment type, work location, salary, proposed joining date, and key conditions.",
                "Verbal discussions do not constitute a final employment offer and must not be treated as authorization to begin work."
            ],
            "Background verification": [
                "Background verification must be conducted through the company's approved process and only with the candidate's required consent.",
                "Verification may include identity, employment history, education, professional credentials, and other checks relevant to the role.",
                "A discrepancy must be reviewed by HR before an adverse decision is made and the candidate must be given an opportunity to clarify material information.",
                "Sensitive verification records must be stored in restricted HR systems and accessed only by authorized personnel."
            ],
            "Joining and induction": [
                "The hiring manager must submit equipment, system access, and workspace requests at least five business days before the joining date.",
                "New employees must complete identity verification, payroll details, emergency contact information, and required employment documentation.",
                "Mandatory security and workplace conduct training should be completed within ten business days after joining.",
                "The employee's manager should conduct an induction meeting during the first two working days and agree on initial responsibilities."
            ],
            "Probation review": [
                "The standard probation period for regular new employees is 90 calendar days unless the employment agreement specifies another duration.",
                "The manager should conduct a midpoint check-in near day 45 and a final review before the end of the probation period.",
                "The review must consider role knowledge, work quality, communication, reliability, and completion of agreed onboarding objectives.",
                "HR must record confirmation, extension, or other employment action in the employee record before the probation end date."
            ]
        }
    },
    "performance_appraisal_policy.html": {
        "title": "Performance Management and Appraisal Policy",
        "owner": "People Operations Department",
        "sections": {
            "Performance planning": [
                "Managers and employees should agree on measurable objectives within 30 calendar days after the beginning of each performance cycle.",
                "Objectives should identify expected outcomes, measurement criteria, target dates, dependencies, and the employee's level of responsibility.",
                "A performance plan should normally contain three to five primary objectives and may include role-specific development goals.",
                "Objectives that change materially because of business priorities must be updated with the employee and recorded in the performance system."
            ],
            "Ongoing feedback": [
                "Managers should hold a one-to-one performance discussion at least once every calendar month.",
                "Discussions should cover progress, blockers, workload, collaboration, development needs, and agreed next actions.",
                "Employees may record their perspective and request support through the performance platform.",
                "Material performance concerns should be communicated promptly rather than being introduced for the first time in the annual appraisal."
            ],
            "Review cycles": [
                "The company conducts formal performance reviews twice each calendar year, normally in June and December.",
                "Employees must complete their self-assessment within ten business days after the review window opens.",
                "Managers should submit ratings and supporting examples within 15 business days after receiving completed employee self-assessments.",
                "Department calibration meetings review consistency across teams and must retain documented reasons for material rating changes."
            ],
            "Performance ratings": [
                "A rating of 5 means exceptional performance substantially exceeded the agreed objectives and demonstrated sustained wider impact.",
                "A rating of 4 means performance exceeded most agreed objectives with strong quality, ownership, and consistency.",
                "A rating of 3 means performance met the agreed requirements and demonstrated the expected level of role competence.",
                "A rating of 2 means some important objectives were not met and a focused improvement plan may be required; a rating of 1 indicates substantially below expectations."
            ],
            "Improvement plans": [
                "A formal performance improvement plan should identify specific gaps, measurable expectations, support actions, and a review period.",
                "The normal improvement plan duration is 60 calendar days, although HR may approve a different period based on role requirements.",
                "The manager should schedule progress reviews at least every two weeks and record factual examples of progress or continuing concerns.",
                "Employees may submit written comments on the plan and request HR clarification about the review process."
            ],
            "Promotion and recognition": [
                "Promotion recommendations must demonstrate sustained performance at the next role level rather than tenure alone.",
                "Managers must provide evidence of increased responsibility, impact, technical capability, and collaboration before submitting a promotion request.",
                "Promotion decisions require department leadership and HR approval, with compensation review before the effective date.",
                "Recognition awards must be based on documented contributions and must follow the relevant departmental award approval limits."
            ]
        }
    },
    "vendor_management_policy.html": {
        "title": "Third-Party Vendor Management Policy",
        "owner": "Procurement and Risk Management",
        "sections": {
            "Vendor onboarding": [
                "A new vendor must complete the approved onboarding process before receiving a purchase order or access to company systems.",
                "The onboarding request must include the vendor legal name, business address, tax information, service description, and primary contact.",
                "Procurement verifies business registration and payment details using an independently obtained contact channel.",
                "Vendors that cannot provide required ownership or registration information must be escalated to Procurement Risk."
            ],
            "Risk assessment": [
                "Vendors processing confidential data or connecting to internal systems require an information security assessment before contract signature.",
                "Vendors handling personal data must also complete the privacy questionnaire and accept applicable data processing obligations.",
                "Critical service providers are assessed for financial stability, operational resilience, business continuity, and subcontractor dependencies.",
                "High-risk vendors require written risk acceptance from the accountable business executive and the relevant control owner."
            ],
            "Contract requirements": [
                "Contracts must identify service scope, delivery obligations, fees, service levels, confidentiality, ownership of work, and termination provisions.",
                "Information security and privacy schedules must be attached where the vendor handles company systems or personal data.",
                "Contracts longer than 24 months require a documented renewal and performance review before the next renewal decision.",
                "Employees must not accept revised commercial terms directly from vendors without Procurement and Legal review."
            ],
            "Service monitoring": [
                "Business owners should review critical vendor performance at least quarterly against documented service levels.",
                "Review records must include incidents, missed service levels, unresolved corrective actions, major changes, and upcoming renewals.",
                "A repeated service-level failure over two consecutive review periods must be escalated to the vendor relationship owner.",
                "Material changes in vendor ownership, hosting location, subcontractors, or service delivery must be reported for risk reassessment."
            ],
            "Vendor access": [
                "Vendor system accounts must be individually assigned, time-bound, and limited to the specific access required for contracted work.",
                "Remote vendor access must use company approved authentication and monitoring controls.",
                "Vendor accounts must be disabled within one business day after contract completion or removal of the access requirement.",
                "Vendors must not share credentials or permit unauthorized subcontractors to use company-issued accounts."
            ],
            "Exit and records": [
                "At termination, the business owner must coordinate return or secure deletion of company information held by the vendor.",
                "Vendor access credentials, badges, equipment, and temporary accounts must be revoked or recovered as part of the exit checklist.",
                "Procurement retains the contract, risk assessment, performance reviews, approvals, and exit confirmation for seven years.",
                "Any unresolved vendor security incident must remain tracked until the accountable risk owner approves closure."
            ]
        }
    },
    "business_continuity_policy.html": {
        "title": "Business Continuity and Disaster Recovery Policy",
        "owner": "Enterprise Risk and IT Operations",
        "sections": {
            "Continuity planning": [
                "Each department must maintain a business continuity plan identifying essential services, critical suppliers, key personnel, and operational dependencies.",
                "Plans must describe how the department will operate during loss of office access, technology outage, utility disruption, or unavailable suppliers.",
                "Department heads must assign a primary continuity coordinator and an alternate coordinator.",
                "Continuity plans must be reviewed at least annually and whenever a material operational or technology change occurs."
            ],
            "Service priorities": [
                "Critical customer-facing services should have a documented recovery time objective of four hours or less.",
                "Important internal services should normally have a recovery time objective of 24 hours or less.",
                "Recovery point objectives must specify the maximum acceptable period of data loss for each critical system.",
                "System owners must document dependencies and recovery priorities so that infrastructure and application services are restored in a workable sequence."
            ],
            "Incident activation": [
                "A department head or designated continuity coordinator may request activation of the departmental continuity plan when normal operations are materially disrupted.",
                "Major incidents affecting multiple departments must be escalated to the Enterprise Incident Manager.",
                "The activation record must state the incident time, affected services, initial impact, decision owner, and communication channel.",
                "Employees must follow incident instructions and avoid publishing unapproved statements about the cause or impact of the disruption."
            ],
            "Backup and recovery": [
                "Critical business data must be backed up according to the system's approved recovery point objective.",
                "Backup copies must be protected from unauthorized alteration and access using controls appropriate to their data classification.",
                "Recovery tests for critical systems must be performed at least twice per year and documented with the achieved recovery time.",
                "A backup is not considered successfully tested until the system owner verifies that required data can be restored and used."
            ],
            "Communication": [
                "The incident communications lead maintains the official status update and ensures departments receive consistent information.",
                "During a major disruption, internal status updates should be issued at least every two hours while material service impacts continue.",
                "Customer notifications must be reviewed by Legal, Communications, and the accountable business owner before distribution.",
                "Employees must use the designated emergency communication channel when corporate email or collaboration systems are unavailable."
            ],
            "Exercises and improvements": [
                "Departments must participate in at least one continuity exercise each calendar year.",
                "Exercise reports should capture the scenario, participants, decisions, recovery times, observed gaps, and corrective action owners.",
                "Corrective actions must be assigned a due date and tracked until closure by the continuity coordinator.",
                "Lessons from real incidents must be incorporated into the relevant continuity plan after an appropriate review."
            ]
        }
    },
    "workplace_safety_policy.html": {
        "title": "Workplace Health and Safety Policy",
        "owner": "Facilities and People Operations",
        "sections": {
            "General responsibilities": [
                "Employees must follow posted safety instructions, use equipment only for its intended purpose, and report hazards to Facilities.",
                "Managers are responsible for identifying workplace risks and ensuring that employees receive role-appropriate safety information.",
                "Visitors and contractors must be informed of relevant site safety requirements and accompanied where access restrictions apply.",
                "Employees must not block emergency exits, fire equipment, electrical panels, or designated evacuation routes."
            ],
            "Hazard reporting": [
                "An employee who discovers a hazard must report it to Facilities as soon as reasonably possible and should isolate the area when safe to do so.",
                "A high-risk hazard involving immediate danger must be reported to site security or emergency services without waiting for a routine Facilities ticket.",
                "Hazard reports should include the location, time, description, people potentially affected, and any immediate protective action taken.",
                "Facilities must record the report, assign an owner, determine the risk level, and document the corrective action."
            ],
            "Accident and injury reporting": [
                "Workplace injuries, near misses, and property damage must be reported to the manager and Facilities within one hour of discovery.",
                "Serious injuries requiring emergency assistance must first be referred to local emergency services.",
                "Incident reports should identify witnesses, location, time, equipment involved, and factual observations without assigning blame.",
                "Managers must preserve relevant evidence and cooperate with an authorized incident investigation."
            ],
            "Emergency procedures": [
                "Employees must follow the building evacuation alarm and proceed to the designated assembly point using the nearest safe exit.",
                "Employees must not use elevators during a fire evacuation unless the building's approved emergency procedure specifically permits it.",
                "Fire wardens coordinate floor sweeps when safe and report missing or unaccounted-for persons to the incident lead.",
                "Employees must not re-enter an evacuated building until the authorized emergency lead confirms that it is safe."
            ],
            "Ergonomics and workstation safety": [
                "Employees should adjust chair height, monitor position, keyboard placement, and lighting to support a safe workstation posture.",
                "Employees who experience recurring discomfort should request an ergonomic assessment through Facilities or occupational health.",
                "Electrical cables must be routed safely and damaged adapters or power strips must be removed from service.",
                "Heavy equipment must be moved using approved handling practices or assistance rather than unsafe lifting."
            ],
            "Training and reviews": [
                "New employees must complete basic workplace safety induction within ten business days after joining.",
                "Employees assigned to laboratory, workshop, or facilities tasks must complete additional role-specific safety training before performing those activities.",
                "Facilities must review emergency contact information and evacuation arrangements at least annually.",
                "Safety inspections should occur quarterly and resulting corrective actions must be tracked to completion."
            ]
        }
    }
}


def render_document(filename, policy):
    sections = []

    for heading, rules in policy["sections"].items():
        paragraphs = "\n".join(
            f"<li>{escape(rule)}</li>" for rule in rules
        )
        sections.append(
            f"<section><h2>{escape(heading)}</h2>"
            f"<ul>{paragraphs}</ul></section>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{escape(policy['title'])}</title>
<meta name="description" content="Synthetic Contoso portfolio policy document">
</head>
<body>
<header>
<h1>{escape(policy['title'])}</h1>
<p>Contoso Ltd. | Synthetic sample document for portfolio testing</p>
<p>Document owner: {escape(policy['owner'])}</p>
<p>Version: 1.0 | Effective date: November 1, 2026</p>
<p>Classification: Internal Use</p>
</header>
<main>
<p>This fictional policy defines responsibilities, controls, approval
requirements, records, and operating procedures for Contoso Ltd.
It is synthetic portfolio data and does not represent a real company policy.</p>
{''.join(sections)}
</main>
<footer>
<p>End of document: {escape(filename)}</p>
</footer>
</body>
</html>
"""


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for filename, policy in POLICIES.items():
        path = OUTPUT_DIR / filename
        path.write_text(
            render_document(filename, policy),
            encoding="utf-8",
        )
        print("Created:", path)

    print("\nDocuments created:", len(POLICIES))
    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
