$(document).ready(function() {
    $('#prediction-input').on('input', function() {
        let span = $('<span></span>').css({
            fontSize: $(this).css('font-size'),
            fontFamily: $(this).css('font-family'),
            fontWeight: $(this).css('font-weight'),
            letterSpacing: $(this).css('letter-spacing'),
            whiteSpace: 'pre',
            visibility: 'hidden',
            position: 'absolute',
        }).appendTo('body');

        span.text($(this).val() || $(this).attr('placeholder'));

        const newWidth = span.width() + 20;
        
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
        this.style.width = newWidth + 'px';

        span.remove();
    });

    let predictionInput = $(".input-prediction");
    let updatedRotation;

    function updateDisplayMessage(message, fontSize) {
        $(".displayed-prediction").html(message);
        $(".displayed-prediction").css({'font-size': fontSize});
    }

    $('.predict-button').click(function() {
        let inputValue = predictionInput.val().trim();
        
        if (inputValue === '') {
            updateDisplayMessage("Invalid input.<br>Please enter some text.", '1em');
            $(".predictionbox-score").css({"transform": "rotate(-45deg)"});
            $(".needle").css({"transform": "translateX(-50%) translateY(0) rotate(-90deg)", "transform-origin": "bottom"});
            return;
        }

        $.ajax({
            type: "POST",
            url: "/predict",
            contentType: "application/json",
            data: JSON.stringify({ input_text: inputValue }),
            success: function(response) {
                console.log(response);
                const suicidePercentage = response.suicide_percentage;
                const predictedClass = response.predicted_class;
                updatedRotation = Math.round(suicidePercentage * 180 / 100) - 45;
                needleRotation = Math.round(suicidePercentage * 180 / 100) - 90;
                updateDisplayMessage("Suicide Percentage: " + suicidePercentage.toFixed(2) + "%", '1em');
                $(".displayed-prediction").append("<br>Predicted Class: " + predictedClass);
                $(".predictionbox-score").css({"transform": "rotate(" + updatedRotation + "deg)"});
                $(".needle").css({"transform": "translateX(-50%) translateY(0) rotate(" + needleRotation + "deg)"});
            },
            error: function() {
                $(".displayed-prediction").append("<br>Error processing the prediction.");
            }
        });
    });
});
