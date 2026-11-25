"""
sample_data.py — Built-in sample job descriptions.

One realistic job description per resume category in the dataset. They power
the "load a sample job" picker in the app and serve as labelled queries for the
JD → resume ranking benchmark in ``evaluation.py`` (a resume is considered
relevant to a job when it belongs to the job's category).
"""

SAMPLE_JOBS: dict[str, dict[str, str]] = {
    "IT Systems Administrator": {
        "category": "INFORMATION-TECHNOLOGY",
        "text": (
            "We are looking for an IT Systems Administrator to keep our infrastructure secure and "
            "running smoothly. You will administer Windows Server and Linux environments, manage "
            "Active Directory users and group policies, maintain LAN/WAN networks, firewalls and VPNs, "
            "and resolve help desk tickets for hardware and software issues. Responsibilities include "
            "patch management, backups and disaster recovery, virtualization with VMware, monitoring "
            "system performance and documenting procedures. Experience with Office 365, SQL Server, "
            "PowerShell scripting and ITIL practices is preferred. CompTIA, CCNA or MCSE certification "
            "is a plus."
        ),
    },
    "HR Generalist": {
        "category": "HR",
        "text": (
            "We are seeking an HR Generalist to support the full employee lifecycle. Responsibilities "
            "include full-cycle recruiting and onboarding, administering benefits and payroll, "
            "maintaining HRIS records, advising managers on employee relations and performance "
            "management, and ensuring compliance with labor laws, FMLA and EEO regulations. You will "
            "coordinate training programs, handle employee inquiries, conduct exit interviews and "
            "support compensation reviews. Bachelor's degree in Human Resources or a related field and "
            "PHR or SHRM-CP certification preferred."
        ),
    },
    "Staff Accountant": {
        "category": "ACCOUNTANT",
        "text": (
            "Staff Accountant needed to manage the general ledger, accounts payable and accounts "
            "receivable. You will prepare journal entries, perform bank and balance sheet "
            "reconciliations, support the month-end and year-end close, and prepare financial "
            "statements in accordance with GAAP. Additional duties include fixed asset accounting, "
            "accruals, sales tax filings and assisting external auditors. Proficiency in QuickBooks, "
            "Excel (pivot tables, VLOOKUP) and ERP systems is required. Bachelor's degree in "
            "Accounting; CPA candidates preferred."
        ),
    },
    "Sous Chef": {
        "category": "CHEF",
        "text": (
            "Seeking an experienced Sous Chef to lead our kitchen team alongside the Executive Chef. "
            "Responsibilities include menu development, food preparation and plating, supervising and "
            "training line cooks, maintaining food safety and sanitation standards, managing inventory "
            "and ordering, controlling food and labor costs, and ensuring consistent quality during "
            "high-volume service. Culinary degree and ServSafe certification preferred, with "
            "experience in fine dining, banquets or catering."
        ),
    },
    "Elementary School Teacher": {
        "category": "TEACHER",
        "text": (
            "Our school is hiring an Elementary School Teacher to plan and deliver engaging, "
            "standards-aligned lessons in reading, writing, math and science. The teacher will "
            "differentiate instruction for diverse learners, assess student progress, manage the "
            "classroom positively, communicate with parents and collaborate with colleagues on "
            "curriculum development. Experience with IEPs, small-group instruction and classroom "
            "technology is valued. Bachelor's degree in Education and a valid state teaching "
            "certification required."
        ),
    },
    "Sales Account Executive": {
        "category": "SALES",
        "text": (
            "We are hiring a Sales Account Executive to grow revenue in an assigned territory. You "
            "will prospect new customers, qualify leads, deliver product demonstrations, negotiate "
            "pricing and close deals, while building long-term client relationships and upselling "
            "existing accounts. Responsibilities include managing the sales pipeline in Salesforce "
            "CRM, forecasting, meeting monthly quotas and representing the company at trade shows. A "
            "proven track record of exceeding sales targets and excellent negotiation and customer "
            "service skills are required."
        ),
    },
    "Graphic Designer": {
        "category": "DESIGNER",
        "text": (
            "Creative Graphic Designer wanted to produce visual concepts for print and digital media. "
            "You will design logos, brochures, packaging, social media graphics, web layouts and "
            "presentations that follow brand guidelines. Responsibilities include developing concepts "
            "from creative briefs, typography and layout, photo editing, preparing print-ready files "
            "and collaborating with marketing teams. Expert knowledge of Adobe Photoshop, Illustrator "
            "and InDesign is required; experience with Figma, UI/UX design and motion graphics is a "
            "plus. Portfolio required."
        ),
    },
    "Personal Trainer": {
        "category": "FITNESS",
        "text": (
            "We are looking for a certified Personal Trainer to coach members toward their health and "
            "fitness goals. You will conduct fitness assessments, design individualized exercise "
            "programs, lead one-on-one and group training sessions including strength, cardio, HIIT "
            "and flexibility classes, and provide nutrition guidance. Responsibilities include "
            "demonstrating proper exercise technique, tracking client progress, promoting gym "
            "memberships and maintaining equipment safety. NASM, ACE or ACSM certification and "
            "CPR/AED certification required."
        ),
    },
    "Registered Nurse": {
        "category": "HEALTHCARE",
        "text": (
            "Registered Nurse needed for a busy medical-surgical unit. You will assess patients, "
            "administer medications and treatments, monitor vital signs, develop and implement nursing "
            "care plans, and document care accurately in the electronic health record. "
            "Responsibilities include educating patients and families, coordinating with physicians "
            "and the interdisciplinary team, following infection control and HIPAA requirements, and "
            "responding to emergencies. Active RN license and BLS/ACLS certification required; "
            "hospital clinical experience preferred."
        ),
    },
    "Relationship Banker": {
        "category": "BANKING",
        "text": (
            "Our bank is hiring a Relationship Banker to deliver exceptional service to retail and "
            "small business clients. You will open deposit accounts, process loan and mortgage "
            "applications, cross-sell banking products and services, resolve customer issues and "
            "build a portfolio of client relationships. Responsibilities include cash handling, branch "
            "operations, meeting sales goals and ensuring compliance with BSA/AML, KYC and bank "
            "policies. Prior experience as a teller, personal banker or in financial services is "
            "preferred."
        ),
    },
    "Construction Project Manager": {
        "category": "CONSTRUCTION",
        "text": (
            "Construction Project Manager required to oversee commercial building projects from "
            "preconstruction through closeout. Duties include managing budgets and schedules, "
            "preparing cost estimates and bids, coordinating subcontractors and suppliers, reviewing "
            "blueprints and specifications, processing RFIs, submittals and change orders, and "
            "enforcing OSHA safety standards on the job site. Experience with Procore, MS Project or "
            "Primavera scheduling and a degree in Construction Management or Civil Engineering are "
            "preferred."
        ),
    },
    "Aircraft Maintenance Technician": {
        "category": "AVIATION",
        "text": (
            "We are seeking an Aircraft Maintenance Technician to inspect, troubleshoot and repair "
            "aircraft systems. Responsibilities include performing scheduled and unscheduled "
            "maintenance on airframes, engines, hydraulics, avionics and landing gear, completing "
            "inspections in accordance with FAA regulations, maintaining accurate maintenance records "
            "and logbooks, and ensuring airworthiness and flight safety. FAA Airframe and Powerplant "
            "(A&P) license required; military aviation experience is a plus."
        ),
    },
    "Financial Analyst": {
        "category": "FINANCE",
        "text": (
            "Financial Analyst wanted to support budgeting, forecasting and strategic decision making. "
            "You will build financial models, analyze variances between actual and budgeted results, "
            "prepare monthly management reports and dashboards, evaluate investment opportunities and "
            "support the annual planning process. The role requires strong Excel and financial "
            "modeling skills, experience with ERP systems such as SAP or Oracle, and knowledge of "
            "corporate finance, cash flow analysis and valuation. Bachelor's degree in Finance or "
            "Economics; CFA progress is a plus."
        ),
    },
    "Mechanical Engineer": {
        "category": "ENGINEERING",
        "text": (
            "Mechanical Engineer needed to design and develop mechanical components and systems. You "
            "will create 3D models and drawings in SolidWorks or AutoCAD, perform engineering "
            "calculations, tolerance analysis and FEA simulations, build and test prototypes, and "
            "support manufacturing with process improvements. Responsibilities include preparing "
            "technical documentation, managing engineering change orders and collaborating with "
            "cross-functional teams on product development. Bachelor's degree in Mechanical "
            "Engineering required; knowledge of GD&T and lean manufacturing preferred."
        ),
    },
    "Legal Associate": {
        "category": "ADVOCATE",
        "text": (
            "Law firm seeks a Legal Associate to support attorneys in litigation and corporate "
            "matters. Responsibilities include legal research and writing, drafting pleadings, "
            "motions, contracts and briefs, preparing discovery documents, managing case files, "
            "interviewing clients and witnesses, and attending court hearings. Knowledge of civil "
            "procedure, Westlaw or LexisNexis, and excellent written advocacy skills are required. "
            "Juris Doctor or paralegal certification preferred."
        ),
    },
    "Digital Marketing Specialist": {
        "category": "DIGITAL-MEDIA",
        "text": (
            "Digital Marketing Specialist wanted to plan and execute online campaigns across social "
            "media, search and email. You will create digital content, manage social media channels, "
            "run SEO and SEM (Google Ads) campaigns, produce video and multimedia content, analyze "
            "performance with Google Analytics, and optimize engagement and conversion rates. "
            "Experience with content management systems such as WordPress, Adobe Creative Suite, "
            "email marketing platforms and video editing is preferred."
        ),
    },
    "Public Relations Specialist": {
        "category": "PUBLIC-RELATIONS",
        "text": (
            "We are hiring a Public Relations Specialist to manage our brand reputation and media "
            "presence. You will write press releases, pitch stories to journalists, build media "
            "relations, coordinate press events and interviews, manage crisis communications, and "
            "develop internal and external communication strategies. Responsibilities include "
            "monitoring media coverage, managing social media messaging, preparing speeches and "
            "community outreach. Excellent writing skills and a degree in Public Relations, "
            "Communications or Journalism required."
        ),
    },
    "Business Development Manager": {
        "category": "BUSINESS-DEVELOPMENT",
        "text": (
            "Business Development Manager needed to identify and secure new growth opportunities. You "
            "will research target markets, generate leads, build strategic partnerships, develop "
            "proposals and pitch to prospective clients, negotiate contracts and manage key accounts. "
            "Responsibilities include creating business plans, tracking market trends and "
            "competitors, forecasting revenue and collaborating with marketing and product teams. "
            "Proven experience in B2B sales or business development, strong networking and "
            "presentation skills required."
        ),
    },
    "Management Consultant": {
        "category": "CONSULTANT",
        "text": (
            "Management Consultant sought to help clients solve complex business problems. You will "
            "analyze client operations, gather requirements through stakeholder interviews, develop "
            "recommendations for process improvement and strategy, build business cases and present "
            "findings to executives. Responsibilities include project management, data analysis, "
            "change management and implementation support across industries. Experience with "
            "consulting engagements, Excel and PowerPoint modeling, and strong problem-solving and "
            "client-facing communication skills required. MBA preferred."
        ),
    },
    "Farm Operations Manager": {
        "category": "AGRICULTURE",
        "text": (
            "We are seeking a Farm Operations Manager to oversee crop production and farm operations. "
            "Responsibilities include planning planting and harvesting schedules, managing "
            "irrigation, soil fertility, pest and weed control, operating and maintaining farm "
            "equipment, supervising farm workers, and ensuring compliance with agricultural and "
            "environmental regulations. Knowledge of agronomy, livestock management, sustainable "
            "farming practices and agricultural record keeping is preferred. Degree in Agriculture, "
            "Agronomy or Animal Science a plus."
        ),
    },
    "Apparel Merchandiser": {
        "category": "APPAREL",
        "text": (
            "Apparel company seeks a Merchandiser to drive product assortment and sales for our "
            "clothing lines. You will analyze sales trends, plan seasonal assortments, collaborate "
            "with designers on product development, work with vendors and factories on sourcing, "
            "costing and production timelines, and coordinate visual merchandising in retail stores. "
            "Knowledge of fashion trends, textiles and fabrics, garment construction, sizing and fit, "
            "and retail buying is required. Experience with inventory planning and Excel is preferred."
        ),
    },
    "Arts Program Coordinator": {
        "category": "ARTS",
        "text": (
            "Arts organization is hiring an Arts Program Coordinator to plan and deliver exhibitions, "
            "performances and community arts education programs. Responsibilities include "
            "coordinating artists and instructors, curating gallery exhibitions, managing event "
            "logistics and budgets, writing grant applications, marketing programs and engaging donors "
            "and volunteers. Background in fine arts, art history, theater or music, with experience "
            "in nonprofit arts administration, is preferred."
        ),
    },
    "Automotive Service Technician": {
        "category": "AUTOMOBILE",
        "text": (
            "Automotive Service Technician needed to diagnose, maintain and repair cars and light "
            "trucks. Duties include performing engine diagnostics, brake, suspension, steering and "
            "transmission repairs, electrical system troubleshooting, oil changes and preventive "
            "maintenance, using diagnostic scan tools and following manufacturer service procedures. "
            "You will document repairs on work orders and explain service needs to customers. ASE "
            "certification preferred; dealership experience and own tools a plus."
        ),
    },
    "Customer Service Representative (BPO)": {
        "category": "BPO",
        "text": (
            "BPO contact center hiring Customer Service Representatives to handle inbound and "
            "outbound calls, emails and chats for our clients. You will resolve customer inquiries and "
            "complaints, process orders, troubleshoot account and billing issues, document "
            "interactions in the CRM, and meet service level, average handle time and quality "
            "targets. Strong communication skills, typing speed, patience and the ability to work "
            "rotating shifts in a call center environment are required. Prior BPO or call center "
            "experience preferred."
        ),
    },
}
