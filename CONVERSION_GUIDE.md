# Guide: Converting FINAL_REPORT.md to DOCX

This guide provides multiple methods to convert the Markdown report to a properly formatted Word document for Gymnasiearbete submission.

---

## ⭐ Recommended Method: Pandoc (Best Quality)

Pandoc is the most powerful tool for Markdown → DOCX conversion with proper formatting.

### Installation

**Windows (PowerShell as Administrator):**
```powershell
# Option 1: Using Chocolatey (recommended)
choco install pandoc

# Option 2: Using Scoop
scoop install pandoc

# Option 3: Download installer from https://pandoc.org/installing.html
```

**Verify installation:**
```bash
pandoc --version
```

### Basic Conversion

```bash
pandoc report\FINAL_REPORT.md -o report\FINAL_REPORT.docx
```

### Advanced Conversion (Recommended)

For better formatting with numbered sections and table of contents:

```bash
pandoc report\README.md ^
    -o report\README.docx ^
    --from markdown ^
    --to docx ^
    --number-sections ^
    --toc ^
    --toc-depth=3 ^
    --metadata title="readme" ^
    --metadata author="Axel Jönsson" ^
    --standalone ^
    --resource-path=report_assets/figures
```

**Or simply run the batch file:**
```bash
convert_to_docx_pandoc.bat
```

### Fine-tuning with Reference Document

For even better control, create a reference document with your desired styles:

1. Create a basic DOCX with desired formatting (fonts, spacing, heading styles)
2. Save as `report_template.docx`
3. Convert with reference:

```bash
pandoc report\FINAL_REPORT.md ^
    -o report\FINAL_REPORT.docx ^
    --reference-doc=report_template.docx ^
    --number-sections ^
    --toc
```

---

## Method 2: Python Script (python-docx)

If you prefer programmatic control:

### Installation

```bash
pip install python-docx
```

### Run the Conversion Script

```bash
python convert_to_docx.py
```

This script:
- Sets correct fonts (Calibri 12pt body, 14pt headings)
- Handles headings with proper levels
- Converts tables
- Inserts figures with captions
- Maintains bullet points and numbered lists

**Pros**: Full control over formatting  
**Cons**: More manual adjustments needed in Word afterward

---

## Method 3: Online Converter (Quick but Limited)

For a quick conversion without installing software:

1. Go to one of these sites:
   - https://cloudconvert.com/md-to-docx
   - https://convertio.co/md-docx/
   - https://www.zamzar.com/convert/md-to-docx/

2. Upload `report/FINAL_REPORT.md`

3. Download the DOCX

**Pros**: No installation needed  
**Cons**: Limited formatting control, figures may not embed properly

---

## Method 4: Manual (Microsoft Word)

If you prefer full manual control:

### Steps:

1. Open Microsoft Word
2. Create new blank document
3. Open `report/FINAL_REPORT.md` in a text editor
4. Copy sections one by one and paste into Word
5. Apply styles manually:
   - Heading 1: Calibri 14pt bold (for "1. Inledning")
   - Heading 2: Calibri 12pt bold (for "1.1 Syfte och mål")
   - Body text: Calibri 12pt
6. Insert figures from `report_assets/figures/` manually

**Pros**: Complete control  
**Cons**: Time-consuming, error-prone

---

## Post-Conversion Checklist

After converting with any method, open the DOCX in Word and verify:

### Formatting
- [ ] Font: Calibri (or Times New Roman) throughout
- [ ] Body text: 12pt
- [ ] Headings: 14pt for level 1, 12pt bold for level 2
- [ ] Margins: 2.5 cm all sides
- [ ] Line spacing: 1.15 or 1.5 (Word default is fine)

### Structure
- [ ] Title page with:
  - Report title (centered, 18pt)
  - Your name (centered)
  - Supervisor name (centered)
- [ ] Sammanfattning on separate page (max 1 page)
- [ ] Abstract on separate page
- [ ] Table of contents (auto-generated):
  - Right-click → Update Field (to refresh page numbers)

### Content
- [ ] All 8 figures inserted and visible
- [ ] Figure captions below each figure
- [ ] Tables formatted with borders
- [ ] References section properly formatted
- [ ] Page numbers in footer (Insert → Page Number)

### Figures
For each figure, verify:
- [ ] Image displays correctly (not broken link)
- [ ] Caption below image (e.g., "Figur 1: Elliptisk omloppsbana...")
- [ ] Centered on page
- [ ] Appropriate size (6 inches wide is good default)

If figures are missing after conversion:
1. Insert manually: Insert → Pictures
2. Select from `report_assets/figures/`
3. Add caption: References → Insert Caption

### Table of Contents

To generate an automatic table of contents:

1. Place cursor where TOC should appear (after Abstract)
2. Go to **References** tab
3. Click **Table of Contents** → Choose style
4. Update later: Right-click TOC → **Update Field** → **Update entire table**

### Page Numbers

1. Insert → Page Number → Bottom of Page → Plain Number 3 (centered)
2. First page (title) often has no number:
   - Double-click header/footer area
   - Check "Different First Page"

---

## Common Issues and Fixes

### Issue: Figures Don't Appear

**Solution 1**: Make sure `--resource-path=report_assets/figures` is in your Pandoc command

**Solution 2**: Convert figure references from:
```markdown
*(Se `report_assets/figures/fig1.png`)*
```

To standard Markdown image syntax:
```markdown
![Figur 1: Elliptisk omloppsbana](report_assets/figures/fig1_elliptical_orbit_rk4.png)
```

**Solution 3**: Insert manually in Word after conversion

### Issue: Swedish Characters (å, ä, ö) Look Wrong

**Solution**: Make sure you're saving/opening with UTF-8 encoding:

```bash
pandoc report\FINAL_REPORT.md -o report\FINAL_REPORT.docx --from markdown+smart --to docx
```

In python-docx, the script already handles UTF-8.

### Issue: Section Numbers Wrong

After conversion, update numbering in Word:
1. Click any heading
2. Home → Multilevel List → Define New Multilevel List
3. Set level 1 as "1.", level 2 as "1.1", etc.

### Issue: Code Blocks Look Ugly

Code blocks in the report (showing the Python implementation) should be:
- Monospace font (Consolas or Courier New)
- 10pt size
- Indented 0.5 inches
- Gray background (optional)

To fix in Word:
1. Select code block
2. Font → Consolas, 10pt
3. Paragraph → Indent left by 1.27 cm

### Issue: Equations Not Rendering

The report uses UTF-8 math symbols (μ, ε, ²) which should display fine. If they don't:
- Change font to one with good Unicode support (Calibri works)
- For complex equations, consider using Word's Equation Editor

---

## Recommended Workflow

**For best results, I recommend:**

1. **Use Pandoc** with the advanced command (or run `convert_to_docx_pandoc.bat`)
2. Open result in Word
3. Fix these manually (takes 5-10 minutes):
   - Add page break before each major section (Insert → Page Break)
   - Verify all figures are inserted correctly
   - Generate table of contents (References → Table of Contents)
   - Add page numbers
   - Add supervisor name on title page
   - Check that Swedish characters display correctly
   - Review and update any formatting inconsistencies

4. Save final version as `FINAL_REPORT_submission.docx`

---

## Alternative: Export from VSCode

If you're using Visual Studio Code with Markdown extensions:

1. Install extension: "Markdown PDF" or "Markdown to DOCX"
2. Open `FINAL_REPORT.md`
3. Right-click → "Markdown: Export (docx)"

**Note**: Results vary by extension, Pandoc is still recommended.

---

## Questions?

**Q: Which method should I use?**  
A: Pandoc (Method 1) for best balance of automation and quality.

**Q: Will figures be embedded or linked?**  
A: Pandoc embeds them. If using python-docx script, they're embedded. Online converters vary.

**Q: What if I don't have admin rights to install Pandoc?**  
A: Use the python-docx script (Method 2) or online converter (Method 3).

**Q: How do I handle the mathematics (μ, ε)?**  
A: UTF-8 symbols work fine in Word with Calibri font. For complex equations, use Word's Equation Editor.

**Q: Can I use Google Docs instead of Word?**  
A: Yes, but export to DOCX afterward. Google Docs can import Markdown files directly.

---

## Final Recommendation

**For submission-ready document:**

```bash
# 1. Convert with Pandoc
pandoc report\FINAL_REPORT.md -o report\FINAL_REPORT.docx --number-sections --toc --standalone

# 2. Open in Word

# 3. Make these manual adjustments (5 min):
#    - Insert page breaks before sections
#    - Update table of contents
#    - Add page numbers
#    - Insert supervisor name
#    - Verify figures

# 4. Save and submit!
```

Total time: < 10 minutes for a polished, professional document.
