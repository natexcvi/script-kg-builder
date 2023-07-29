package knowledgegraph

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
)

type RelationEmbedder struct {
	client         *http.Client
	apiToken       string
	embeddingModel string
	tokensUsed     int
}

func NewRelationEmbedder(client *http.Client, apiToken string) *RelationEmbedder {
	return &RelationEmbedder{
		client:         client,
		apiToken:       apiToken,
		embeddingModel: "text-embedding-ada-002",
	}
}

type embeddingRequest struct {
	Input string `json:"input"`
	Model string `json:"model"`
}

type embeddingResponse struct {
	Data []struct {
		Embedding []float64 `json:"embedding"`
	} `json:"data"`
	Usage struct {
		TotalTokens int `json:"total_tokens"`
	} `json:"usage"`
}

func (e *RelationEmbedder) EmbedRelation(relation string) ([]float64, error) {
	embedReq := embeddingRequest{
		Input: relation,
		Model: e.embeddingModel,
	}
	marshaledReq, err := json.Marshal(embedReq)
	if err != nil {
		return []float64{}, fmt.Errorf("failed to marshal embedding request: %v", err)
	}

	req, err := http.NewRequest("POST", "https://api.openai.com/v1/embeddings", ioutil.NopCloser(bytes.NewReader(marshaledReq)))
	if err != nil {
		return []float64{}, fmt.Errorf("failed to create embedding request: %v", err)
	}

	req.Header.Add("Authorization", fmt.Sprintf("Bearer %s", e.apiToken))
	req.Header.Add("Content-Type", "application/json")

	resp, err := e.client.Do(req)
	if err != nil {
		return []float64{}, fmt.Errorf("failed to send embedding request: %v", err)
	}
	defer resp.Body.Close()
	var embedResp embeddingResponse
	if err := json.NewDecoder(resp.Body).Decode(&embedResp); err != nil {
		return []float64{}, fmt.Errorf("failed to decode embedding response: %v", err)
	}
	e.tokensUsed += embedResp.Usage.TotalTokens
	if len(embedResp.Data) == 0 {
		return []float64{}, fmt.Errorf("empty embedding response: %+v", embedResp)
	}
	return embedResp.Data[0].Embedding, nil
}

func (e *RelationEmbedder) TokensUsed() int {
	return e.tokensUsed
}
