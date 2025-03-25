class ResultsDisplay:
    def __init__(self, results: Dict, image: Image):
        self.results = results
        self.image = image

    def render(self):
        """Render the complete results display."""
        tabs = st.tabs(["📊 Probability", "🔬 Visualization", "📋 Details"])
        
        with tabs[0]:
            self.render_probability_tab()
        with tabs[1]:
            self.render_visualization_tab()
        with tabs[2]:
            self.render_details_tab()

    def render_probability_tab(self):
        pass 