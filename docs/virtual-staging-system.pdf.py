"""Generate the Virtual Staging System Overview PDF — Empty Rooms Only."""
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

output_path = "/Users/ryanhaugland/virtual-staging/docs/virtual-staging-system-overview.pdf"

doc = SimpleDocTemplate(
    output_path,
    pagesize=letter,
    leftMargin=1 * inch,
    rightMargin=1 * inch,
    topMargin=1 * inch,
    bottomMargin=1 * inch,
)

# Styles
title_style = ParagraphStyle(
    "Title", fontName="Helvetica-Bold", fontSize=20, spaceAfter=20, leading=24
)
h1_style = ParagraphStyle(
    "H1", fontName="Helvetica-Bold", fontSize=14, spaceBefore=20, spaceAfter=10, leading=18
)
h2_style = ParagraphStyle(
    "H2", fontName="Helvetica-Bold", fontSize=12, spaceBefore=14, spaceAfter=6, leading=16
)
body_style = ParagraphStyle(
    "Body", fontName="Helvetica", fontSize=10, spaceAfter=8, leading=14, alignment=TA_LEFT
)
bullet_style = ParagraphStyle(
    "Bullet", fontName="Helvetica", fontSize=10, spaceAfter=4, leading=14,
    leftIndent=20, bulletIndent=10, bulletFontName="Helvetica",
)
numbered_style = ParagraphStyle(
    "Numbered", fontName="Helvetica", fontSize=10, spaceAfter=6, leading=14,
    leftIndent=20,
)

story = []

# Title
story.append(Paragraph("Virtual Staging System Overview", title_style))
story.append(Paragraph("Empty Room Staging with Google Gemini", body_style))
story.append(Spacer(1, 10))

# --- How It Works ---
story.append(Paragraph("How It Works", h1_style))

story.append(Paragraph(
    "The system takes a photo of an empty room and adds furniture using Google Gemini 3 Pro Image Preview "
    "(gemini-3-pro-image-preview). There is no layering, no post-processing, and no second model pass. "
    "Gemini receives the original photo plus a detailed text prompt and returns a single new image "
    "with furniture digitally placed into the empty space. One API call per photo.",
    body_style
))

# --- Consistency ---
story.append(Paragraph("How Multiple Photos Stay Consistent", h1_style))

story.append(Paragraph(
    "When a user uploads multiple photos of the same property, three techniques work together "
    "to ensure the staged furniture looks the same across all photos:",
    body_style
))

story.append(Paragraph(
    "<b>1. Same Seed.</b> One random seed is generated per job and reused for every photo in that job. "
    "This reduces Gemini's randomness across calls, making outputs more deterministic.",
    numbered_style
))
story.append(Paragraph(
    "<b>2. Predefined Furniture Catalogs.</b> Instead of saying 'modern furniture,' the system describes "
    "the exact sofa, chairs, coffee table, rug, and art piece down to material, color, arm style, "
    "leg type, and cushion shape. Every photo in the job receives the identical furniture description.",
    numbered_style
))
story.append(Paragraph(
    "<b>3. Explicit Quantity Enforcement.</b> The prompt specifies exact counts: 'exactly ONE sofa, "
    "exactly TWO accent chairs.' Without this, the furniture count drifts between photos.",
    numbered_style
))

# --- User Inputs ---
story.append(Paragraph("User Inputs", h1_style))

story.append(Paragraph(
    "<bullet>&bull;</bullet> <b>Room Type</b> (living, bedroom, dining, kitchen, office, bathroom) "
    "-- selects the furniture list and room name used in the prompt.",
    bullet_style
))
story.append(Paragraph(
    "<bullet>&bull;</bullet> <b>Style</b> (modern, traditional, scandinavian, luxury, coastal, "
    "midcentury modern, farmhouse, industrial, minimalist, contemporary, boho, rustic) "
    "-- selects the detailed furniture description with exact colors, materials, and shapes.",
    bullet_style
))
story.append(Paragraph(
    "<bullet>&bull;</bullet> <b>Photos</b> (1-4 images) "
    "-- each photo gets the same prompt and the same seed.",
    bullet_style
))

# --- The Prompt ---
story.append(Paragraph("The Prompt Structure", h1_style))

story.append(Paragraph(
    "The prompt sent to Gemini is structured as four numbered rules. "
    "The order matters -- preservation rules come first so Gemini anchors on them before reading the furniture instructions.",
    body_style
))

story.append(Paragraph("RULE 1 -- PRESERVE EVERYTHING", h2_style))
story.append(Paragraph(
    "Do not remove, replace, or alter anything already in this photo. "
    "Every countertop stays exactly as-is -- same material, color, shape, fully visible, "
    "fully opaque, fully solid. Do not make countertops transparent or translucent. "
    "Every cabinet, sink, appliance, fixture stays exactly as-is. "
    "Walls, floors, ceiling, windows, doors, trim, molding, vents, outlets -- all unchanged. "
    "The camera has NOT moved -- same angle, same height, same position, same lens, same field of view. "
    "Do not zoom, crop, rotate, or shift the viewpoint at all. "
    "The output must be pixel-aligned with the input photo.",
    body_style
))

story.append(Paragraph("RULE 2 -- ADD FURNITURE", h2_style))
story.append(Paragraph(
    "Edit this photo of an empty [room type]. "
    "Place furniture ONLY in empty floor space, never where countertops or cabinets are. "
    "Add exactly: [furniture list with explicit counts]. "
    "Example for living room: exactly one 3-seat sofa, exactly two accent chairs, "
    "one coffee table, two side tables, one area rug, two table lamps, throw pillows, "
    "and one piece of wall art.",
    body_style
))

story.append(Paragraph("RULE 3 -- EXACT STYLE", h2_style))
story.append(Paragraph(
    "This is where the predefined furniture catalog is inserted. Example for Modern style: "
    "'Furniture: charcoal gray linen 3-seat track-arm sofa with square cushions, "
    "two matching charcoal gray linen square-back armchairs with chrome legs, "
    "round white lacquer coffee table, chrome arc floor lamp, "
    "black and white geometric area rug, abstract black and white art print on wall.' "
    "Followed by: Use EXACTLY the colors and materials specified -- do not adapt them to match the room.",
    body_style
))

story.append(Paragraph("RULE 4 -- EXACT COUNT", h2_style))
story.append(Paragraph(
    "Place exactly ONE sofa and exactly TWO accent chairs. No more, no fewer. "
    "The sofa must be the exact same style described above in every photo. "
    "Photorealistic real estate photograph.",
    body_style
))

# --- Available Styles ---
story.append(Paragraph("Available Design Styles", h1_style))

styles_list = [
    ("Modern", "Charcoal gray linen track-arm sofa, chrome legs, black and white geometric rug, abstract art."),
    ("Traditional", "Navy blue tufted velvet rolled-arm sofa, burgundy wingback chairs, oriental rug, oil painting."),
    ("Scandinavian", "White linen sofa with rounded arms, pale gray chairs, light oak coffee table, ceramic vases."),
    ("Luxury", "Emerald green velvet Chesterfield sofa, navy barrel-back chairs, white marble and gold coffee table."),
    ("Coastal", "White slipcovered sofa, rattan armchairs, driftwood coffee table, seagrass rug, nautical art."),
    ("Mid-Century Modern", "Mustard yellow sofa with tapered walnut legs, olive green chairs, geometric rug."),
    ("Farmhouse", "Cream linen slipcovered sofa, distressed white oak coffee table, jute rug, botanical print."),
    ("Industrial", "Dark brown leather Chesterfield, club armchairs with rivets, iron and reclaimed wood table."),
    ("Minimalist", "White low-profile platform sofa, single armchair, ash wood coffee table, one art print."),
    ("Contemporary", "Taupe boucle curved sofa, brass-leg chairs, white marble coffee table, earth-toned abstract art."),
    ("Boho", "Terracotta linen floor sofa, rattan peacock chairs, kilim rug, macrame wall hanging."),
    ("Rustic", "Cognac brown leather rolled-arm sofa, club chairs, dark pine coffee table, cowhide rug."),
]

for name, desc in styles_list:
    story.append(Paragraph(f"<bullet>&bull;</bullet> <b>{name}:</b> {desc}", bullet_style))

# --- Best Practices ---
story.append(Paragraph("Prompt Engineering Best Practices", h1_style))

practices = [
    ("Preservation rules first.", "Countertop and structure protection goes at the top of the prompt, not the bottom. Gemini deprioritizes rules that appear later in the prompt."),
    ("Numbered rules.", "Using 'RULE 1', 'RULE 2' gives Gemini clear structure to follow and reduces rule-skipping."),
    ("Furniture-only style descriptions.", "Style prompts describe ONLY furniture (sofa, chairs, rug, art). Never mention room surfaces like 'shiplap walls' or 'exposed brick' -- that causes Gemini to alter the room structure."),
    ("Exact material, color, and shape.", "'Charcoal gray linen 3-seat track-arm sofa with square cushions' not 'modern sofa.' The more specific the description, the more consistent the output across multiple photos."),
    ("Explicit color lock.", "'Use EXACTLY the colors specified -- do not adapt them to match the room.' Without this instruction, Gemini shifts furniture colors to complement the room's existing palette."),
    ("Explicit quantity lock.", "'Exactly ONE sofa, exactly TWO accent chairs, no more, no fewer.' Without this, the number of furniture pieces varies from photo to photo."),
    ("Anti-transparency rule.", "'Fully opaque, fully solid, do not make countertops transparent.' This prevents the furniture style from visually bleeding into existing room surfaces."),
    ("Anti-zoom rule.", "'The camera has NOT moved -- same angle, same height, same position, same lens, same field of view. The output must be pixel-aligned with the input.' Gemini tends to reframe or subtly zoom without this."),
    ("Placement constraint.", "'Place furniture ONLY in empty floor space, never where countertops or cabinets are.' This prevents furniture from overlapping existing room elements."),
    ("Same seed across photos.", "One random seed per job reduces randomness so the same prompt produces more visually similar results on different photos of the same property."),
]

for i, (title, desc) in enumerate(practices, 1):
    story.append(Paragraph(f"<b>{i}. {title}</b> {desc}", numbered_style))

# --- API Details ---
story.append(Paragraph("API Details", h1_style))

story.append(Paragraph(
    "<bullet>&bull;</bullet> <b>Endpoint:</b> generativelanguage.googleapis.com/v1beta/models/gemini-3-pro-image-preview:generateContent",
    bullet_style
))
story.append(Paragraph(
    "<bullet>&bull;</bullet> <b>Input:</b> Base64-encoded image + text prompt, sent as a JSON POST request.",
    bullet_style
))
story.append(Paragraph(
    "<bullet>&bull;</bullet> <b>Generation Config:</b> responseModalities set to IMAGE, seed set to the job's shared integer.",
    bullet_style
))
story.append(Paragraph(
    "<bullet>&bull;</bullet> <b>Output:</b> Base64-encoded image returned inline in the JSON response.",
    bullet_style
))
story.append(Paragraph(
    "<bullet>&bull;</bullet> <b>Timeout:</b> 300 seconds per request, with up to 3 retries using 10-second and 20-second backoff for 503, 429, and 500 errors.",
    bullet_style
))

# --- Tech Stack ---
story.append(Paragraph("Tech Stack", h1_style))

story.append(Paragraph(
    "<bullet>&bull;</bullet> <b>Backend:</b> Python, FastAPI, Uvicorn",
    bullet_style
))
story.append(Paragraph(
    "<bullet>&bull;</bullet> <b>Frontend:</b> Vanilla HTML/CSS/JavaScript",
    bullet_style
))
story.append(Paragraph(
    "<bullet>&bull;</bullet> <b>AI Model:</b> Google Gemini 3 Pro Image Preview (direct REST API, no SDK required)",
    bullet_style
))
story.append(Paragraph(
    "<bullet>&bull;</bullet> <b>Tunnel:</b> ngrok for public access during development",
    bullet_style
))

doc.build(story)
print(f"PDF saved to {output_path}")
