class NewsService {
    constructor() {
        this.newsBaseUrl = "https://gnews.io/api/v4/search";
        this.newsApiKey = "736f3f2a4626a28db8712ed791a11578";
    }

    // Helper to assign priority based on article content
    assignPriority(article) {
        const emergencyKeywords = ['urgent', 'emergency', 'disaster', 'outbreak', 'crisis'];
        const warningKeywords = ['warning', 'alert', 'advisory', 'concern', 'risk'];
        
        const text = (article.title + ' ' + article.description).toLowerCase();
        
        if (emergencyKeywords.some(keyword => text.includes(keyword))) {
            return 'high';
        } else if (warningKeywords.some(keyword => text.includes(keyword))) {
            return 'medium';
        }
        return 'low';
    }

    async getNews(country = "us", category = null) {
        const params = new URLSearchParams({
            token: this.newsApiKey,
            lang: "en",
            country: country,
            max: "10",
            q: "agriculture"
        });

        try {
            const response = await fetch(`${this.newsBaseUrl}?${params}`);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            return {
                articles: (data.articles || []).map(article => ({
                    title: article.title,
                    description: article.description,
                    url: article.url,
                    publishedAt: article.publishedAt,
                    source: { name: article.source?.name },
                    priority: this.assignPriority(article)
                }))
            };
        } catch (error) {
            console.error("Failed to fetch news:", error);
            throw error;
        }
    }
}