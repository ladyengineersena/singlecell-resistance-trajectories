# Ethics and Data Protection Guidelines

## Overview

This repository and associated code are designed for **research purposes only**. The following guidelines are **mandatory** for any use of this pipeline with patient data.

## Absolute Requirements

### 1. Institutional Review Board (IRB) Approval

- **IRB approval is REQUIRED** before using any patient data
- All research involving human subjects must comply with local regulations (e.g., HIPAA in the US, GDPR in the EU)
- Data Transfer Agreements (DTA) must be in place when sharing data between institutions

### 2. Patient Consent

- **Written informed consent** is required from all patients whose data is used
- Consent must explicitly cover:
  - Use of biological samples for research
  - Genomic/transcriptomic analysis
  - Data sharing (if applicable)
  - Publication of results

### 3. Data Protection

#### Protected Health Information (PHI) / Personal Identifiable Information (PII)

- **NEVER commit real patient data to this repository**
- **NEVER upload patient data to public repositories** (including GitHub)
- All patient identifiers must be removed or anonymized:
  - Names, dates of birth, medical record numbers
  - Exact dates (use "days since" relative dates)
  - Geographic identifiers smaller than state/region
  - Any other information that could identify individuals

#### Anonymization Standards

- Use study IDs instead of patient names
- Replace dates with relative timepoints (e.g., "days since diagnosis")
- Remove or generalize demographic information if it could identify individuals
- Use k-anonymity principles when reporting results

### 4. Repository Content

This repository contains **ONLY**:
- ✅ Code and analysis pipelines
- ✅ Synthetic/demo data generators
- ✅ Documentation and tutorials
- ✅ Example outputs (anonymized)

This repository does **NOT** contain:
- ❌ Real patient data
- ❌ PHI/PII
- ❌ Clinical records
- ❌ Sequencing data from patients

### 5. Model Limitations and Disclaimers

#### Research Use Only

- Models are **decision-support tools**, not definitive treatment recommendations
- All predictions are **hypotheses** that require biological and clinical validation
- **Human-in-the-loop** is mandatory for any clinical application

#### Model Outputs

- Model predictions should be clearly labeled as "research predictions"
- Include confidence intervals and uncertainty estimates
- Document model limitations and potential biases

### 6. Fairness and Equity

#### Demographic Justice

- **Mandatory reporting** of model performance across:
  - Demographic subgroups (age, sex, race/ethnicity)
  - Clinical subgroups (disease stage, prior treatments)
  - Genetic subgroups (if applicable)
- Identify and address performance disparities
- Document any biases in training data

#### Subgroup Analysis

- Report metrics (AUC, precision, recall) for each subgroup
- Include sample sizes for each subgroup
- Discuss limitations when subgroups are underrepresented

## Data Sharing Guidelines

### Internal Sharing

- Use secure, encrypted storage for patient data
- Limit access to authorized personnel only
- Maintain audit logs of data access

### External Sharing

- Only share anonymized, aggregated data
- Use data sharing agreements
- Consider controlled-access repositories (e.g., dbGaP) for genomic data

### Publication

- Never publish identifiable patient information
- Use aggregate statistics and anonymized visualizations
- Follow journal guidelines for data availability

## Best Practices

### 1. Data Management

- Store patient data separately from code repositories
- Use version control for code, not for patient data
- Implement data retention policies

### 2. Documentation

- Document all data transformations
- Maintain data dictionaries
- Record any data quality issues

### 3. Validation

- Validate predictions against known outcomes when possible
- Perform sensitivity analyses
- Document model assumptions

### 4. Collaboration

- Ensure all collaborators understand ethical requirements
- Provide training on data protection
- Regular ethics reviews for ongoing projects

## Compliance Checklist

Before using this pipeline with patient data, verify:

- [ ] IRB approval obtained
- [ ] Patient consent forms collected
- [ ] Data Transfer Agreement (if applicable) signed
- [ ] All PHI/PII removed or anonymized
- [ ] Data stored securely (not in public repositories)
- [ ] Access controls implemented
- [ ] Model limitations documented
- [ ] Subgroup performance analyzed
- [ ] Human-in-the-loop process defined
- [ ] Ethics review completed

## Reporting Issues

If you discover any potential ethical issues or data breaches:

1. **Immediately** notify your institution's IRB and data protection officer
2. Document the issue
3. Take corrective action
4. Update procedures to prevent recurrence

## Resources

- [HIPAA Privacy Rule](https://www.hhs.gov/hipaa/index.html) (US)
- [GDPR](https://gdpr.eu/) (EU)
- [TCGA Data Use Agreement](https://www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga/history/policies)
- [Single Cell Portal Data Policies](https://singlecell.broadinstitute.org/single_cell)

## Contact

For ethical concerns or questions, contact:
- Your institution's IRB
- Data protection officer
- Research ethics committee

---

**Remember**: When in doubt, prioritize patient privacy and data protection over convenience or speed.

