Dropzone.autoDiscover = false;

function init() {

    let dz = new Dropzone("#dropzone", {
        url: "/",
        maxFiles: 1,
        addRemoveLinks: true,
        dictDefaultMessage: "Some Message",
        autoProcessQueue: false
    });

    dz.on("addedfile", function () {
        if (dz.files[1] != null) {
            dz.removeFile(dz.files[0]);
        }
    });

    dz.on("complete", function (file) {

        $.post("http://127.0.0.1:5000/classify_image", {
            image_data: file.dataURL
        }, function (data, status) {

            console.log(data);

            // ERROR CASE
            if (!data || data.length == 0) {
                $("#predictionCard").hide();
                $("#classTable").hide();
                $("#error").show();
                return;
            }

            let match = null;
            let bestScore = -1;

            // FIND BEST MATCH
            for (let i = 0; i < data.length; i++) {
                let maxScore = Math.max(...data[i].class_probability);

                if (maxScore > bestScore) {
                    bestScore = maxScore;
                    match = data[i];
                }
            }

            if (match) {

                $("#error").hide();
                $("#predictionCard").show();
                $("#classTable").show();

                // Clean name
                let predictedName = match.class.replaceAll("_", " ");

                $("#predictionName").text(predictedName);

                $("#predictionConfidence").text(
                    "Confidence: " + bestScore.toFixed(2) + "%"
                );

                // Set image
                $("#predictionImage").attr(
                    "src",
                    "./images/" + match.class + ".jpg"
                );

                // Update table
                let classDictionary = match.class_dictionary;

                for (let personName in classDictionary) {

                    let index = classDictionary[personName];
                    let probability = match.class_probability[index];

                    $("#score_" + personName).html(
                        probability.toFixed(2) + "%"
                    );
                }
            }
        });

    });

    // BUTTON CLICK
    $("#submitBtn").on('click', function () {
        dz.processQueue();
    });
}

$(document).ready(function () {

    console.log("ready!");

    $("#error").hide();
    $("#predictionCard").hide();
    $("#classTable").hide();

    init();
});