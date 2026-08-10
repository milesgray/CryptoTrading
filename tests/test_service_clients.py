from cryptotrading.client import (
    ArtifactService,
    ExchangePriceClient,
    PriceServerClient,
    SimulatedPriceClient,
    EmbedServiceClient,
    RetrievalServiceClient,
    PredictServiceClient,
    PressureServiceClient,
    JepaServiceClient,
    SentimentServiceClient,
    ServeServiceClient,
    TradeServiceClient,
    TrainServiceClient
)

def test_clients_imports_and_initialization():
    embed = EmbedServiceClient(base_url="http://localhost:8000")
    assert embed.base_url == "http://localhost:8000"

    retrieval = RetrievalServiceClient(base_url="http://localhost:8001")
    assert retrieval.base_url == "http://localhost:8001"

    predict = PredictServiceClient(base_url="http://localhost:8002")
    assert predict.base_url == "http://localhost:8002"

    pressure = PressureServiceClient(base_url="http://localhost:8003")
    assert pressure.base_url == "http://localhost:8003"

    jepa = JepaServiceClient(base_url="http://localhost:8004")
    assert jepa.base_url == "http://localhost:8004"

    sentiment = SentimentServiceClient(base_url="http://localhost:8005")
    assert sentiment.base_url == "http://localhost:8005"

    serve = ServeServiceClient(base_url="http://localhost:8006")
    assert serve.base_url == "http://localhost:8006"

    trade = TradeServiceClient(base_url="http://localhost:8007")
    assert trade.base_url == "http://localhost:8007"

    train = TrainServiceClient(base_url="http://localhost:8008")
    assert train.base_url == "http://localhost:8008"
